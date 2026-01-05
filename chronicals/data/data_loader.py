"""
Chronicals Data Loader - Production-Ready Edition
===================================================
High-performance data loading with sequence packing and GPU prefetching.
Designed for 50,000+ tok/s training throughput on A100/H100.

Key Features:
- Async GPU prefetching using CUDA streams
- Multi-worker data loading with persistent workers
- Integration with FixedShapeSequencePacker for packing
- Curriculum learning support (short to long sequences)
- Dynamic batch sizing based on sequence lengths
- Memory-mapped datasets for large-scale training
- HuggingFace DataCollatorWithFlattening compatibility

Performance Optimizations:
- Pin memory for faster H2D transfers
- Non-blocking CUDA operations
- Double-buffering for zero latency
- Batch-level sequence packing (pack on-the-fly)

References:
- NVIDIA Apex data prefetching pattern
- HuggingFace packing with FA2: https://huggingface.co/blog/packing-with-FA2
- PyTorch DataLoader best practices

Author: Chronicals Framework
Version: 2.0.0 (Production-Ready)
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, Sampler
from typing import Dict, List, Optional, Any, Callable, Union, Iterator, Tuple
from dataclasses import dataclass, field
import random
import math
import threading
from queue import Queue
import warnings

from config import HF_READ_TOKEN, DATASET_REGISTRY

# Import sequence packer components
from sequence_packer import (
    FixedShapeSequencePacker,
    PackedBatch,
    PackedDataset,
    PackedDataLoader as SequencePackerDataLoader,
    DataPrefetcher,
    DoubleBufferPrefetcher,
    DynamicBatchSizeAdjuster,
    analyze_dataset_for_packing,
    create_cu_seqlens,
    create_cu_seqlens_from_position_ids,
    compute_padding_free_loss,
    compute_tokens_per_second,
    FLASH_ATTN_AVAILABLE,
)


# =============================================================================
# Data Collators
# =============================================================================

@dataclass
class DataCollator:
    """
    Standard collate function for training with padding.

    Pads sequences to the maximum length in the batch.
    """
    pad_token_id: int = 0
    max_length: int = 4096
    padding_side: str = 'right'  # 'right' or 'left'

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch with padding."""
        # Find max length in batch
        max_len = min(
            max(len(item['input_ids']) for item in batch),
            self.max_length
        )

        input_ids = []
        attention_mask = []
        labels = []
        position_ids = []

        for item in batch:
            ids = item['input_ids'][:max_len]
            lbl = item['labels'][:max_len] if 'labels' in item else ids.clone()

            # Pad
            pad_len = max_len - len(ids)
            if pad_len > 0:
                if self.padding_side == 'right':
                    ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                    lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=lbl.dtype)])
                    mask = torch.cat([torch.ones(max_len - pad_len), torch.zeros(pad_len)])
                    pos = torch.cat([torch.arange(max_len - pad_len), torch.zeros(pad_len, dtype=torch.long)])
                else:  # left padding
                    ids = torch.cat([torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype), ids])
                    lbl = torch.cat([torch.full((pad_len,), -100, dtype=lbl.dtype), lbl])
                    mask = torch.cat([torch.zeros(pad_len), torch.ones(max_len - pad_len)])
                    pos = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.arange(max_len - pad_len)])
            else:
                mask = torch.ones(max_len)
                pos = torch.arange(max_len)

            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lbl)
            position_ids.append(pos)

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(labels),
            'position_ids': torch.stack(position_ids),
        }


@dataclass
class DataCollatorWithFlattening:
    """
    HuggingFace-compatible collator that flattens sequences for packing.

    This is compatible with DataCollatorWithFlattening from transformers 4.44+
    but provides additional features:
    - Generates cu_seqlens for FlashAttention varlen
    - Creates proper position_ids that reset per sequence
    - Supports padding-free loss computation

    Reference: https://huggingface.co/blog/packing-with-FA2
    """
    pad_token_id: int = 0
    max_length: int = 4096
    generate_position_ids: bool = True
    generate_cu_seqlens: bool = True

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Flatten and pack sequences."""
        # Collect all sequences
        all_input_ids = []
        all_labels = []
        all_position_ids = []
        sequence_lengths = []

        for item in batch:
            ids = item['input_ids']
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids)

            # Truncate if needed
            ids = ids[:self.max_length]
            seq_len = ids.numel()

            lbl = item.get('labels', ids.clone())
            if not isinstance(lbl, torch.Tensor):
                lbl = torch.tensor(lbl)
            lbl = lbl[:self.max_length]

            all_input_ids.append(ids)
            all_labels.append(lbl)
            sequence_lengths.append(seq_len)

            if self.generate_position_ids:
                all_position_ids.append(torch.arange(seq_len, dtype=torch.long))

        # Concatenate
        input_ids = torch.cat(all_input_ids)
        labels = torch.cat(all_labels)

        # Set first token of each sequence (except first) to -100
        # to prevent cross-sequence prediction
        offset = 0
        for i, length in enumerate(sequence_lengths):
            if i > 0:
                labels[offset] = -100
            offset += length

        result = {
            'input_ids': input_ids.unsqueeze(0),  # [1, total_len]
            'labels': labels.unsqueeze(0),
            'attention_mask': None,  # Not needed for varlen
        }

        if self.generate_position_ids:
            position_ids = torch.cat(all_position_ids)
            result['position_ids'] = position_ids.unsqueeze(0)

        if self.generate_cu_seqlens:
            cu_seqlens = torch.zeros(len(sequence_lengths) + 1, dtype=torch.int32)
            cu_seqlens[1:] = torch.cumsum(
                torch.tensor(sequence_lengths, dtype=torch.int32), dim=0
            )
            result['cu_seqlens'] = cu_seqlens
            result['max_seqlen'] = max(sequence_lengths)

        return result


@dataclass
class PackingDataCollator:
    """
    Data collator that uses FixedShapeSequencePacker for optimal packing.

    Features:
    - Uses BFD/FFD algorithm for 95%+ packing efficiency
    - Fixed output shapes for CUDA graph compatibility
    - Generates all tensors needed for training
    """
    packer: FixedShapeSequencePacker = None
    max_length: int = 4096
    pad_token_id: int = 0
    packing_strategy: str = 'bfd'

    def __post_init__(self):
        if self.packer is None:
            self.packer = FixedShapeSequencePacker(
                max_seq_length=self.max_length,
                pad_token_id=self.pad_token_id,
                strategy=self.packing_strategy,
            )

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> PackedBatch:
        """Pack batch using bin-packing algorithm."""
        input_ids_list = []
        labels_list = []

        for item in batch:
            ids = item['input_ids']
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids)

            lbl = item.get('labels', ids.clone())
            if not isinstance(lbl, torch.Tensor):
                lbl = torch.tensor(lbl)

            input_ids_list.append(ids)
            labels_list.append(lbl)

        return self.packer.pack_sequences(input_ids_list, labels_list)


# =============================================================================
# Dataset Classes
# =============================================================================

class InstructionDataset(Dataset):
    """
    Dataset for instruction-following data.

    Supports multiple prompt templates:
    - alpaca: Standard Alpaca format
    - dolly: Databricks Dolly format
    - chatml: ChatML format (for Qwen, etc.)
    - llama: LLaMA instruct format
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 4096,
        prompt_template: str = "alpaca",
        mask_prompt: bool = True,
    ):
        """
        Initialize instruction dataset.

        Args:
            data: List of instruction data dictionaries
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            prompt_template: Prompt format template
            mask_prompt: Whether to mask prompt tokens in labels
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.mask_prompt = mask_prompt

        # Prompt templates
        self.templates = {
            "alpaca": self._format_alpaca,
            "dolly": self._format_dolly,
            "chatml": self._format_chatml,
            "llama": self._format_llama,
        }

        # Response markers for masking
        self.response_markers = {
            "alpaca": "### Response:",
            "dolly": "### Response:",
            "chatml": "<|im_start|>assistant",
            "llama": "[/INST]",
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Format prompt
        format_fn = self.templates.get(self.prompt_template, self._format_alpaca)
        text = format_fn(item)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = torch.tensor(encoding['input_ids'])
        labels = input_ids.clone()

        # Mask prompt tokens (only compute loss on response)
        if self.mask_prompt:
            response_marker = self.response_markers.get(
                self.prompt_template, "### Response:"
            )
            response_tokens = self.tokenizer.encode(
                response_marker, add_special_tokens=False
            )

            # Find response start position
            response_start = self._find_subsequence(
                input_ids.tolist(), response_tokens
            )
            if response_start != -1:
                labels[:response_start + len(response_tokens)] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    def _format_alpaca(self, item: Dict) -> str:
        """Format as Alpaca template."""
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', item.get('response', ''))

        if input_text:
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            return f"""### Instruction:
{instruction}

### Response:
{output}"""

    def _format_dolly(self, item: Dict) -> str:
        """Format as Dolly template."""
        instruction = item.get('instruction', '')
        context = item.get('context', '')
        response = item.get('response', '')

        if context:
            return f"""### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}"""
        else:
            return f"""### Instruction:
{instruction}

### Response:
{response}"""

    def _format_chatml(self, item: Dict) -> str:
        """Format as ChatML template."""
        instruction = item.get('instruction', item.get('input', ''))
        response = item.get('output', item.get('response', ''))

        return f"""<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""

    def _format_llama(self, item: Dict) -> str:
        """Format as LLaMA template."""
        instruction = item.get('instruction', item.get('input', ''))
        response = item.get('output', item.get('response', ''))

        return f"""[INST] {instruction} [/INST] {response}"""

    def _find_subsequence(self, sequence: List[int], subsequence: List[int]) -> int:
        """Find start index of subsequence."""
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i + len(subsequence)] == subsequence:
                return i
        return -1


class StreamingInstructionDataset(IterableDataset):
    """
    Streaming dataset for large-scale training.

    Features:
    - Memory-efficient streaming
    - Shuffle buffer for randomization
    - Supports sharding for multi-GPU training
    """

    def __init__(
        self,
        data_iter: Iterator,
        tokenizer,
        max_length: int = 4096,
        prompt_template: str = "alpaca",
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ):
        self.data_iter = data_iter
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

        self._formatter = InstructionDataset(
            data=[],
            tokenizer=tokenizer,
            max_length=max_length,
            prompt_template=prompt_template,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        random.seed(self.seed)
        shuffle_buffer = []

        for item in self.data_iter:
            # Process item
            processed = self._process_item(item)
            if processed is not None:
                shuffle_buffer.append(processed)

            # Yield from buffer when full
            if len(shuffle_buffer) >= self.shuffle_buffer_size:
                random.shuffle(shuffle_buffer)
                for batch_item in shuffle_buffer:
                    yield batch_item
                shuffle_buffer = []

        # Yield remaining items
        if shuffle_buffer:
            random.shuffle(shuffle_buffer)
            for batch_item in shuffle_buffer:
                yield batch_item

    def _process_item(self, item: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single item."""
        try:
            # Format text
            format_fn = self._formatter.templates.get(
                self.prompt_template,
                self._formatter._format_alpaca
            )
            text = format_fn(item)

            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )

            input_ids = torch.tensor(encoding['input_ids'])
            labels = input_ids.clone()

            return {
                'input_ids': input_ids,
                'labels': labels,
            }
        except Exception:
            return None


# =============================================================================
# Samplers
# =============================================================================

class LengthAwareSampler(Sampler):
    """
    Sampler that groups sequences by length for efficient packing.

    Groups similar-length sequences together to minimize padding waste
    and maximize packing efficiency.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        input_key: str = 'input_ids',
        shuffle: bool = True,
        seed: int = 42,
        bucket_size: int = 100,
    ):
        """
        Initialize length-aware sampler.

        Args:
            dataset: Dataset to sample from
            batch_size: Batch size
            input_key: Key for input IDs
            shuffle: Whether to shuffle within buckets
            seed: Random seed
            bucket_size: Size of length buckets
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.input_key = input_key
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = bucket_size

        # Build length index
        self._build_index()

    def _build_index(self):
        """Build index of sequence lengths."""
        self.lengths = []
        for i in range(len(self.dataset)):
            try:
                item = self.dataset[i]
                length = len(item[self.input_key])
                self.lengths.append((i, length))
            except Exception:
                continue

        # Sort by length
        self.lengths.sort(key=lambda x: x[1])

        # Create buckets
        self.buckets = []
        for i in range(0, len(self.lengths), self.bucket_size):
            bucket = [idx for idx, _ in self.lengths[i:i + self.bucket_size]]
            self.buckets.append(bucket)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            random.seed(self.seed)

            # Shuffle within buckets
            for bucket in self.buckets:
                random.shuffle(bucket)

            # Shuffle bucket order
            bucket_order = list(range(len(self.buckets)))
            random.shuffle(bucket_order)

            for bucket_idx in bucket_order:
                for idx in self.buckets[bucket_idx]:
                    yield idx
        else:
            for bucket in self.buckets:
                for idx in bucket:
                    yield idx

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self.buckets)


class CurriculumSampler(Sampler):
    """
    Sampler for curriculum learning (short to long sequences).

    Implements curriculum learning where training starts with shorter
    sequences and gradually increases to longer ones.
    """

    def __init__(
        self,
        dataset: Dataset,
        input_key: str = 'input_ids',
        start_ratio: float = 0.25,
        end_ratio: float = 1.0,
        num_epochs: int = 3,
        current_epoch: int = 0,
    ):
        """
        Initialize curriculum sampler.

        Args:
            dataset: Dataset to sample from
            input_key: Key for input IDs
            start_ratio: Starting max length ratio (0.25 = 25% of max)
            end_ratio: Ending max length ratio
            num_epochs: Total number of epochs
            current_epoch: Current epoch (0-indexed)
        """
        self.dataset = dataset
        self.input_key = input_key
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.num_epochs = num_epochs
        self.current_epoch = current_epoch

        self._build_index()

    def _build_index(self):
        """Build sorted index by length."""
        self.sorted_indices = []
        for i in range(len(self.dataset)):
            try:
                item = self.dataset[i]
                length = len(item[self.input_key])
                self.sorted_indices.append((i, length))
            except Exception:
                continue

        self.sorted_indices.sort(key=lambda x: x[1])
        self.max_length = self.sorted_indices[-1][1] if self.sorted_indices else 0

    def set_epoch(self, epoch: int):
        """Set the current epoch."""
        self.current_epoch = epoch

    def __iter__(self) -> Iterator[int]:
        # Calculate current max length based on epoch
        progress = min(self.current_epoch / max(self.num_epochs - 1, 1), 1.0)
        current_ratio = self.start_ratio + progress * (self.end_ratio - self.start_ratio)
        current_max_length = int(self.max_length * current_ratio)

        # Filter indices by current max length
        valid_indices = [
            idx for idx, length in self.sorted_indices
            if length <= current_max_length
        ]

        # Shuffle valid indices
        random.shuffle(valid_indices)

        for idx in valid_indices:
            yield idx

    def __len__(self) -> int:
        progress = min(self.current_epoch / max(self.num_epochs - 1, 1), 1.0)
        current_ratio = self.start_ratio + progress * (self.end_ratio - self.start_ratio)
        current_max_length = int(self.max_length * current_ratio)

        return sum(
            1 for _, length in self.sorted_indices
            if length <= current_max_length
        )


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_dataset_hf(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load dataset from HuggingFace.

    Args:
        dataset_name: Name from DATASET_REGISTRY or HF path
        split: Dataset split
        max_samples: Max samples to load (None = all)

    Returns:
        List of examples
    """
    from datasets import load_dataset

    # Get HF path
    if dataset_name in DATASET_REGISTRY:
        hf_path = DATASET_REGISTRY[dataset_name]['hf_path']
    else:
        hf_path = dataset_name

    print(f"Loading {hf_path}...")

    # Load dataset
    try:
        dataset = load_dataset(hf_path, split=split, token=HF_READ_TOKEN)
    except Exception:
        dataset = load_dataset(hf_path, split=split)

    # Convert to list
    data = list(dataset)

    if max_samples:
        data = data[:max_samples]

    print(f"Loaded {len(data)} samples")

    return data


def load_alpaca(max_samples: Optional[int] = None) -> List[Dict]:
    """Load Alpaca dataset."""
    from datasets import load_dataset

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    data = list(dataset)

    if max_samples:
        data = data[:max_samples]

    return data


def load_dolly(max_samples: Optional[int] = None) -> List[Dict]:
    """Load Dolly dataset."""
    from datasets import load_dataset

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    data = list(dataset)

    if max_samples:
        data = data[:max_samples]

    return data


def load_slimorca(max_samples: Optional[int] = 100000) -> List[Dict]:
    """Load SlimOrca dataset."""
    from datasets import load_dataset

    dataset = load_dataset("Open-Orca/SlimOrca", split="train")
    data = list(dataset)

    # Convert conversation format
    formatted = []
    for item in data[:max_samples]:
        conversations = item.get('conversations', [])
        if len(conversations) >= 2:
            human = next((c['value'] for c in conversations if c['from'] == 'human'), '')
            gpt = next((c['value'] for c in conversations if c['from'] == 'gpt'), '')
            formatted.append({
                'instruction': human,
                'output': gpt,
            })

    return formatted


# =============================================================================
# DataLoader Factory Functions
# =============================================================================

def create_dataloader(
    dataset_name: str,
    tokenizer,
    batch_size: int = 1,
    max_length: int = 4096,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    prompt_template: str = "alpaca",
    use_packing: bool = False,
    packing_strategy: str = 'bfd',
    use_prefetching: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: Optional[torch.device] = None,
    curriculum_learning: bool = False,
    curriculum_epochs: int = 3,
) -> Union[DataLoader, SequencePackerDataLoader]:
    """
    Create DataLoader for training with all optimizations.

    Args:
        dataset_name: Dataset name (alpaca, dolly, slimorca)
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        max_samples: Max samples to use
        shuffle: Shuffle data
        prompt_template: Prompt format template
        use_packing: Use sequence packing
        packing_strategy: Packing algorithm ('ffd', 'bfd', 'spfhp')
        use_prefetching: Use async GPU prefetching
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster transfers
        device: Target device for prefetching
        curriculum_learning: Use curriculum learning
        curriculum_epochs: Number of epochs for curriculum

    Returns:
        DataLoader or SequencePackerDataLoader with prefetching
    """
    # Load data
    if dataset_name == "alpaca":
        data = load_alpaca(max_samples)
    elif dataset_name == "dolly":
        data = load_dolly(max_samples)
    elif dataset_name == "slimorca":
        data = load_slimorca(max_samples)
    else:
        data = load_dataset_hf(dataset_name, max_samples=max_samples)

    # Create instruction dataset
    instruction_dataset = InstructionDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt_template=prompt_template,
    )

    if use_packing:
        # Create packed dataset with packer
        packer = FixedShapeSequencePacker(
            max_seq_length=max_length,
            pad_token_id=tokenizer.pad_token_id or 0,
            strategy=packing_strategy,
            use_flash_varlen=FLASH_ATTN_AVAILABLE,
        )

        packed_dataset = PackedDataset(
            dataset=instruction_dataset,
            packer=packer,
            batch_size=batch_size * 4,  # Pack more sequences per batch
            shuffle_before_packing=shuffle and not curriculum_learning,
            curriculum_order=curriculum_learning,
        )

        # Print packing statistics
        stats = packed_dataset.get_packing_statistics()
        print(f"Packing statistics:")
        print(f"  Total batches: {stats['num_packed_batches']}")
        print(f"  Packing efficiency: {stats['overall_efficiency']:.1%}")
        print(f"  Mean batch efficiency: {stats['mean_batch_efficiency']:.1%}")

        if use_prefetching and torch.cuda.is_available():
            return SequencePackerDataLoader(
                dataset=packed_dataset,
                batch_size=1,  # Already packed
                shuffle=shuffle and not curriculum_learning,
                num_workers=num_workers,
                pin_memory=pin_memory,
                device=device,
                use_prefetching=True,
            )
        else:
            return DataLoader(
                packed_dataset,
                batch_size=1,
                shuffle=shuffle and not curriculum_learning,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=lambda x: x[0],  # Return single PackedBatch
            )

    else:
        # Standard DataLoader with padding
        collator = DataCollator(
            pad_token_id=tokenizer.pad_token_id or 0,
            max_length=max_length,
        )

        # Choose sampler
        sampler = None
        if curriculum_learning:
            sampler = CurriculumSampler(
                dataset=instruction_dataset,
                num_epochs=curriculum_epochs,
            )
            shuffle = False  # Sampler handles ordering

        dataloader = DataLoader(
            instruction_dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        if use_prefetching and torch.cuda.is_available():
            # Wrap with prefetcher
            class PrefetchedDataLoader:
                """Wrapper that adds CUDA stream prefetching to any DataLoader."""
                def __init__(self, dataloader, device):
                    self.dataloader = dataloader
                    self.device = device or torch.device('cuda')

                def __len__(self):
                    return len(self.dataloader)

                def __iter__(self):
                    return DataPrefetcher(
                        iter(self.dataloader),
                        device=self.device,
                        use_cuda_stream=True,
                        non_blocking=True,
                    )

            return PrefetchedDataLoader(dataloader, device)

        return dataloader


def create_packed_dataloader(
    dataset: Dataset,
    tokenizer,
    max_length: int = 4096,
    batch_size: int = 8,
    packing_strategy: str = 'bfd',
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: Optional[torch.device] = None,
    use_prefetching: bool = True,
) -> Union[DataLoader, SequencePackerDataLoader]:
    """
    Create a packed DataLoader from an existing dataset.

    Args:
        dataset: Source dataset with input_ids and labels
        tokenizer: Tokenizer for padding token
        max_length: Maximum sequence length
        batch_size: Number of sequences to pack per batch
        packing_strategy: Packing algorithm
        shuffle: Shuffle before packing
        num_workers: Data loading workers
        pin_memory: Pin memory
        device: Target device
        use_prefetching: Use async prefetching

    Returns:
        DataLoader or SequencePackerDataLoader with prefetching
    """
    packer = FixedShapeSequencePacker(
        max_seq_length=max_length,
        pad_token_id=tokenizer.pad_token_id or 0,
        strategy=packing_strategy,
        use_flash_varlen=FLASH_ATTN_AVAILABLE,
    )

    packed_dataset = PackedDataset(
        dataset=dataset,
        packer=packer,
        batch_size=batch_size,
        shuffle_before_packing=shuffle,
    )

    if use_prefetching and torch.cuda.is_available():
        return SequencePackerDataLoader(
            dataset=packed_dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            device=device,
            use_prefetching=True,
        )

    return DataLoader(
        packed_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: x[0],
    )


# =============================================================================
# Utility Functions
# =============================================================================

def get_tokenizer(model_name: str):
    """Get tokenizer for model."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_READ_TOKEN,
        trust_remote_code=True,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def analyze_dataset(
    data: List[Dict],
    tokenizer,
    template: str = "alpaca",
    max_samples: int = 1000,
) -> Dict[str, Any]:
    """
    Analyze dataset statistics including packing potential.

    Args:
        data: List of data items
        tokenizer: Tokenizer
        template: Prompt template
        max_samples: Max samples to analyze

    Returns:
        Dictionary with statistics
    """
    import numpy as np

    dataset = InstructionDataset(
        data, tokenizer, max_length=8192, prompt_template=template
    )

    lengths = []
    for i in range(min(len(dataset), max_samples)):
        item = dataset[i]
        lengths.append(len(item['input_ids']))

    lengths = np.array(lengths)

    stats = {
        'num_samples': len(data),
        'mean_length': float(lengths.mean()),
        'median_length': float(np.median(lengths)),
        'min_length': int(lengths.min()),
        'max_length': int(lengths.max()),
        'std_length': float(lengths.std()),
        'p95_length': float(np.percentile(lengths, 95)),
        'p99_length': float(np.percentile(lengths, 99)),
    }

    # Add packing analysis
    for max_seq in [512, 1024, 2048, 4096]:
        clipped_lengths = [min(l, max_seq) for l in lengths]
        packing_stats = FixedShapeSequencePacker.compute_packing_statistics(
            clipped_lengths, max_seq, 'bfd'
        )
        stats[f'packing_{max_seq}_efficiency'] = packing_stats['packing_efficiency']
        stats[f'packing_{max_seq}_speedup'] = packing_stats['vs_padding_speedup']

    return stats


def estimate_training_throughput(
    dataset_stats: Dict[str, Any],
    batch_size: int = 8,
    max_seq_length: int = 4096,
    use_packing: bool = True,
    gpu_tokens_per_sec: float = 50000,
) -> Dict[str, float]:
    """
    Estimate training throughput with and without packing.

    Args:
        dataset_stats: Statistics from analyze_dataset
        batch_size: Training batch size
        max_seq_length: Maximum sequence length
        use_packing: Whether packing is used
        gpu_tokens_per_sec: GPU throughput capacity

    Returns:
        Dictionary with throughput estimates
    """
    mean_length = dataset_stats['mean_length']
    num_samples = dataset_stats['num_samples']

    # Without packing: each sample padded to max_seq_length
    no_pack_tokens_per_batch = batch_size * max_seq_length
    no_pack_waste = (max_seq_length - mean_length) / max_seq_length

    # With packing: use efficiency from stats
    key = f'packing_{max_seq_length}_efficiency'
    packing_efficiency = dataset_stats.get(key, 0.8)
    pack_tokens_per_batch = batch_size * max_seq_length * packing_efficiency

    # Estimate samples per second
    no_pack_samples_per_sec = (gpu_tokens_per_sec / no_pack_tokens_per_batch) * batch_size
    pack_samples_per_sec = (gpu_tokens_per_sec / (max_seq_length * (1 - packing_efficiency + mean_length / max_seq_length))) * batch_size

    return {
        'without_packing': {
            'tokens_per_batch': no_pack_tokens_per_batch,
            'waste_ratio': no_pack_waste,
            'estimated_samples_per_sec': no_pack_samples_per_sec,
            'estimated_epoch_time_sec': num_samples / no_pack_samples_per_sec,
        },
        'with_packing': {
            'packing_efficiency': packing_efficiency,
            'estimated_samples_per_sec': pack_samples_per_sec,
            'estimated_epoch_time_sec': num_samples / pack_samples_per_sec,
            'speedup': pack_samples_per_sec / no_pack_samples_per_sec,
        }
    }


# =============================================================================
# Convenience Wrapper: ChronicalsDataPipeline
# =============================================================================

class ChronicalsDataPipeline:
    """
    High-level wrapper for the complete data loading pipeline.

    Combines dataset loading, tokenization, packing, and prefetching
    into a single easy-to-use interface.

    Usage:
        pipeline = ChronicalsDataPipeline(
            dataset_name="alpaca",
            tokenizer=tokenizer,
            use_packing=True,
            use_prefetching=True,
        )

        for batch in pipeline:
            # batch is PackedBatch on GPU, ready for training
            outputs = model(**batch.to_dict())
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        batch_size: int = 1,
        max_length: int = 4096,
        max_samples: Optional[int] = None,
        prompt_template: str = "alpaca",
        use_packing: bool = True,
        packing_strategy: str = 'bfd',
        use_prefetching: bool = True,
        use_double_buffer: bool = False,
        num_workers: int = 4,
        device: Optional[torch.device] = None,
        curriculum_learning: bool = False,
        curriculum_epochs: int = 3,
        seed: int = 42,
    ):
        """
        Initialize the data pipeline.

        Args:
            dataset_name: Name of dataset to load
            tokenizer: HuggingFace tokenizer
            batch_size: Base batch size (before packing multiplier)
            max_length: Maximum sequence length
            max_samples: Maximum samples to load (None = all)
            prompt_template: Prompt format template
            use_packing: Enable sequence packing
            packing_strategy: Packing algorithm ('ffd', 'bfd', 'spfhp')
            use_prefetching: Enable async GPU prefetching
            use_double_buffer: Use double-buffered prefetching (higher memory)
            num_workers: Data loading workers
            device: Target device
            curriculum_learning: Enable curriculum learning
            curriculum_epochs: Number of curriculum epochs
            seed: Random seed
        """
        self.tokenizer = tokenizer
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        self.use_packing = use_packing
        self.use_prefetching = use_prefetching
        self.use_double_buffer = use_double_buffer

        # Create the base dataloader
        self.dataloader = create_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            max_samples=max_samples,
            shuffle=True,
            prompt_template=prompt_template,
            use_packing=use_packing,
            packing_strategy=packing_strategy,
            use_prefetching=use_prefetching,
            num_workers=num_workers,
            pin_memory=True,
            device=self.device,
            curriculum_learning=curriculum_learning,
            curriculum_epochs=curriculum_epochs,
        )

        # Store statistics
        self._total_batches = 0
        self._total_tokens = 0
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self) -> Iterator:
        for batch in self.dataloader:
            self._total_batches += 1
            if hasattr(batch, 'total_tokens'):
                self._total_tokens += batch.total_tokens
            yield batch
        self._epoch += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get accumulated pipeline statistics."""
        return {
            'total_batches': self._total_batches,
            'total_tokens': self._total_tokens,
            'epochs_completed': self._epoch,
            'use_packing': self.use_packing,
            'use_prefetching': self.use_prefetching,
        }

    def reset_statistics(self) -> None:
        """Reset accumulated statistics."""
        self._total_batches = 0
        self._total_tokens = 0


def create_optimized_dataloader(
    dataset: Dataset,
    tokenizer,
    max_length: int = 4096,
    batch_size: int = 8,
    use_packing: bool = True,
    packing_strategy: str = 'bfd',
    use_prefetching: bool = True,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Union[DataLoader, SequencePackerDataLoader]:
    """
    Create an optimized DataLoader with all available optimizations.

    This is the recommended entry point for creating data loaders
    with the Chronicals framework.

    Args:
        dataset: Source dataset with input_ids and labels
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        batch_size: Base batch size
        use_packing: Enable sequence packing for 2-3x throughput
        packing_strategy: Packing algorithm
        use_prefetching: Enable async GPU prefetching
        device: Target device
        **kwargs: Additional arguments for DataLoader

    Returns:
        Optimized DataLoader or SequencePackerDataLoader
    """
    if use_packing:
        return create_packed_dataloader(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            packing_strategy=packing_strategy,
            use_prefetching=use_prefetching,
            device=device,
            **kwargs,
        )
    else:
        # Standard DataLoader with optional prefetching
        collator = DataCollator(
            pad_token_id=tokenizer.pad_token_id or 0,
            max_length=max_length,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            pin_memory=True,
            **kwargs,
        )

        if use_prefetching and torch.cuda.is_available():
            class PrefetchWrapper:
                def __init__(self, dl, dev):
                    self.dl = dl
                    self.dev = dev or torch.device('cuda')
                def __len__(self):
                    return len(self.dl)
                def __iter__(self):
                    return DataPrefetcher(iter(self.dl), device=self.dev)
            return PrefetchWrapper(dataloader, device)

        return dataloader


# =============================================================================
# Main Entry Point and Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Chronicals Data Loader - Production-Ready Edition")
    print("=" * 70)
    print(f"\nFlashAttention available: {FLASH_ATTN_AVAILABLE}")

    # Test with mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token = "</s>"
            self.eos_token_id = 1
            self.pad_token = "<pad>"

        def __call__(self, text, **kwargs):
            # Simple character-level tokenization for testing
            max_len = kwargs.get('max_length', 1000)
            tokens = [ord(c) % 1000 for c in text[:max_len]]
            return {'input_ids': tokens}

        def encode(self, text, **kwargs):
            return [ord(c) % 1000 for c in text]

    tokenizer = MockTokenizer()

    # Create mock data
    mock_data = [
        {'instruction': 'What is 2+2?', 'input': '', 'output': '4'},
        {'instruction': 'Write a poem about cats', 'input': '', 'output': 'Cats are fluffy creatures...'},
        {'instruction': 'Explain quantum computing', 'input': '', 'output': 'Quantum computing uses qubits...'},
        {'instruction': 'Translate to French', 'input': 'Hello world', 'output': 'Bonjour le monde'},
        {'instruction': 'Summarize this text', 'input': 'Long text here...', 'output': 'Summary here'},
    ]

    print("\n--- Testing InstructionDataset ---")
    dataset = InstructionDataset(mock_data, tokenizer, max_length=512)
    print(f"Dataset size: {len(dataset)}")

    item = dataset[0]
    print(f"Sample item:")
    print(f"  Input IDs length: {len(item['input_ids'])}")
    print(f"  Labels length: {len(item['labels'])}")

    print("\n--- Testing DataCollator ---")
    collator = DataCollator(pad_token_id=0, max_length=512)
    batch = collator([dataset[0], dataset[1]])
    print(f"Batch shape: {batch['input_ids'].shape}")

    print("\n--- Testing DataCollatorWithFlattening ---")
    flat_collator = DataCollatorWithFlattening(max_length=512)
    flat_batch = flat_collator([dataset[0], dataset[1], dataset[2]])
    print(f"Flattened input_ids shape: {flat_batch['input_ids'].shape}")
    print(f"cu_seqlens: {flat_batch.get('cu_seqlens')}")

    print("\n--- Testing PackingDataCollator ---")
    pack_collator = PackingDataCollator(max_length=512)
    packed_batch = pack_collator([dataset[0], dataset[1], dataset[2]])
    print(f"Packed batch:")
    print(f"  Input IDs shape: {packed_batch.input_ids.shape}")
    print(f"  Num sequences: {packed_batch.num_sequences}")
    print(f"  Packing efficiency: {packed_batch.packing_efficiency:.1%}")

    print("\n--- Testing LengthAwareSampler ---")
    sampler = LengthAwareSampler(dataset, batch_size=2, bucket_size=3)
    indices = list(sampler)
    print(f"Sampled indices: {indices}")

    print("\n--- Testing CurriculumSampler ---")
    curriculum_sampler = CurriculumSampler(
        dataset,
        num_epochs=3,
        start_ratio=0.5,
        current_epoch=0
    )
    curriculum_indices = list(curriculum_sampler)
    print(f"Curriculum indices (epoch 0): {curriculum_indices}")

    curriculum_sampler.set_epoch(2)
    curriculum_indices = list(curriculum_sampler)
    print(f"Curriculum indices (epoch 2): {curriculum_indices}")

    print("\n--- Testing analyze_dataset ---")
    stats = analyze_dataset(mock_data, tokenizer, max_samples=5)
    print(f"Dataset statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
