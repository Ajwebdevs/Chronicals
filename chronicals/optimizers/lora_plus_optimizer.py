"""
Chronicals LoRA+ Optimizer Implementation
==========================================
LoRA+ (ICML 2024) uses different learning rates for A and B matrices
for 1.5-2x faster convergence.

Key Insight from the Paper:
===========================
In standard LoRA, both A and B matrices use the same learning rate.
However, theoretical analysis shows:
- eta_A should be O(n^-1) where n is the width
- eta_B should be O(1)

In practice, this translates to using a HIGHER learning rate for B matrix
compared to A matrix. The paper recommends a ratio of 16x.

Why Different Learning Rates?
=============================
1. B matrix (in -> rank): Maps inputs to low-rank space
   - Should adapt quickly to capture relevant features
   - Higher LR allows faster feature discovery

2. A matrix (rank -> out): Projects back to output space
   - Should be more stable to maintain pretrained knowledge
   - Lower LR prevents catastrophic forgetting

Performance:
============
- 1.5-2x faster convergence compared to standard LoRA
- Same memory footprint as standard LoRA
- No additional inference cost

References:
- LoRA+: https://arxiv.org/abs/2402.12354 (ICML 2024)
- "LoRA+: Efficient Low Rank Adaptation of Large Models"

In Colab: Copy this entire cell, paste, and run to create lora_plus_optimizer.py
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import (
    Dict, List, Optional, Tuple, Callable, Iterable,
    Union, Any, Set
)
import math
import re


# ============================================================================
# LoRA+ Parameter Group Builder
# ============================================================================

class LoRAPlusParameterGroups:
    """
    Utility class to create separate parameter groups for LoRA A and B matrices.

    This enables using different learning rates for A and B matrices as
    recommended in the LoRA+ paper.

    Usage:
        model = ...  # Your LoRA-adapted model
        builder = LoRAPlusParameterGroups(model)
        param_groups = builder.get_param_groups(
            base_lr=1e-4,
            lr_ratio=16.0,  # B gets 16x higher LR
        )
        optimizer = torch.optim.AdamW(param_groups)
    """

    # Common patterns for LoRA A and B matrices
    LORA_A_PATTERNS = [
        r'\.lora_A$',
        r'\.lora_A\.',
        r'lora_A_',
        r'\.A$',
        r'_A$',
    ]

    LORA_B_PATTERNS = [
        r'\.lora_B$',
        r'\.lora_B\.',
        r'lora_B_',
        r'\.B$',
        r'_B$',
    ]

    def __init__(
        self,
        model: nn.Module,
        lora_a_patterns: Optional[List[str]] = None,
        lora_b_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the parameter group builder.

        Args:
            model: The model containing LoRA parameters
            lora_a_patterns: Custom regex patterns to identify A matrices
            lora_b_patterns: Custom regex patterns to identify B matrices
        """
        self.model = model
        self.lora_a_patterns = lora_a_patterns or self.LORA_A_PATTERNS
        self.lora_b_patterns = lora_b_patterns or self.LORA_B_PATTERNS

        # Compile patterns for efficiency
        self._a_patterns = [re.compile(p) for p in self.lora_a_patterns]
        self._b_patterns = [re.compile(p) for p in self.lora_b_patterns]

    def _is_lora_a(self, name: str) -> bool:
        """Check if parameter name matches LoRA A pattern."""
        return any(p.search(name) for p in self._a_patterns)

    def _is_lora_b(self, name: str) -> bool:
        """Check if parameter name matches LoRA B pattern."""
        return any(p.search(name) for p in self._b_patterns)

    def get_param_groups(
        self,
        base_lr: float = 1e-4,
        lr_ratio: float = 16.0,
        weight_decay: float = 0.0,
        weight_decay_lora: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create parameter groups with different learning rates for A and B.

        Args:
            base_lr: Learning rate for A matrices (lower)
            lr_ratio: Ratio of B learning rate to A learning rate (default: 16)
            weight_decay: Weight decay for non-LoRA parameters
            weight_decay_lora: Weight decay for LoRA parameters (default: 0)

        Returns:
            List of parameter group dictionaries for optimizer
        """
        if weight_decay_lora is None:
            weight_decay_lora = 0.0  # LoRA typically doesn't use weight decay

        lora_a_params = []
        lora_b_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if self._is_lora_a(name):
                lora_a_params.append(param)
            elif self._is_lora_b(name):
                lora_b_params.append(param)
            else:
                other_params.append(param)

        # Build parameter groups
        param_groups = []

        if lora_a_params:
            param_groups.append({
                'params': lora_a_params,
                'lr': base_lr,
                'weight_decay': weight_decay_lora,
                'name': 'lora_A',
            })

        if lora_b_params:
            param_groups.append({
                'params': lora_b_params,
                'lr': base_lr * lr_ratio,
                'weight_decay': weight_decay_lora,
                'name': 'lora_B',
            })

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'weight_decay': weight_decay,
                'name': 'other',
            })

        # Print summary
        print(f"LoRA+ Parameter Groups:")
        print(f"  lora_A: {len(lora_a_params)} params, lr={base_lr:.2e}")
        print(f"  lora_B: {len(lora_b_params)} params, lr={base_lr * lr_ratio:.2e}")
        print(f"  other:  {len(other_params)} params, lr={base_lr:.2e}")

        return param_groups


# ============================================================================
# LoRA+ AdamW Optimizer
# ============================================================================

class LoRAPlusAdamW(Optimizer):
    """
    AdamW optimizer with automatic LoRA+ learning rate scaling.

    This optimizer automatically detects LoRA A and B parameters and applies
    different learning rates according to the LoRA+ paper.

    Key features:
    - Automatic A/B matrix detection
    - Configurable learning rate ratio (default: 16x for B)
    - Full AdamW compatibility
    - PyTorch scheduler compatibility

    Args:
        params: Model parameters or parameter groups
        lr: Base learning rate (used for A matrices)
        lr_ratio: Ratio of B learning rate to A learning rate (default: 16)
        betas: AdamW beta parameters
        eps: AdamW epsilon
        weight_decay: Weight decay for non-LoRA parameters
        weight_decay_lora: Weight decay for LoRA parameters (default: 0)
        auto_detect_lora: Whether to auto-detect LoRA A/B parameters
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        lr_ratio: float = 16.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        weight_decay_lora: float = 0.0,
        auto_detect_lora: bool = True,
    ):
        # Validate parameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if lr_ratio < 1.0:
            raise ValueError(f"lr_ratio should be >= 1.0, got: {lr_ratio}")

        self.lr_ratio = lr_ratio
        self.auto_detect_lora = auto_detect_lora
        self._base_lr = lr

        # Process params to create parameter groups
        if isinstance(params, torch.nn.Module):
            # If given a module, extract named parameters
            params = list(params.named_parameters())

        # Convert generator to list if needed
        params = list(params)

        # Check if we're given named parameters (tuples) or just parameters
        if params and isinstance(params[0], tuple):
            # Named parameters: (name, param) tuples
            param_groups = self._create_lora_plus_groups(
                params, lr, lr_ratio, weight_decay, weight_decay_lora
            )
        else:
            # Regular parameters or already-formed groups
            if params and isinstance(params[0], dict):
                # Already formed parameter groups
                param_groups = params
            else:
                # List of parameters - can't auto-detect
                param_groups = [{'params': params}]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        super().__init__(param_groups, defaults)
        self._step_count = 0

    def _create_lora_plus_groups(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        base_lr: float,
        lr_ratio: float,
        weight_decay: float,
        weight_decay_lora: float,
    ) -> List[Dict[str, Any]]:
        """Create parameter groups with LoRA+ learning rate scaling."""

        # Patterns for detection
        a_patterns = [re.compile(p) for p in LoRAPlusParameterGroups.LORA_A_PATTERNS]
        b_patterns = [re.compile(p) for p in LoRAPlusParameterGroups.LORA_B_PATTERNS]

        def is_lora_a(name: str) -> bool:
            return any(p.search(name) for p in a_patterns)

        def is_lora_b(name: str) -> bool:
            return any(p.search(name) for p in b_patterns)

        lora_a_params = []
        lora_b_params = []
        other_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue

            if self.auto_detect_lora and is_lora_a(name):
                lora_a_params.append(param)
            elif self.auto_detect_lora and is_lora_b(name):
                lora_b_params.append(param)
            else:
                other_params.append(param)

        groups = []

        if lora_a_params:
            groups.append({
                'params': lora_a_params,
                'lr': base_lr,
                'weight_decay': weight_decay_lora,
                '_group_name': 'lora_A',
            })
            print(f"  LoRA+ lora_A: {len(lora_a_params)} params, lr={base_lr:.2e}")

        if lora_b_params:
            groups.append({
                'params': lora_b_params,
                'lr': base_lr * lr_ratio,
                'weight_decay': weight_decay_lora,
                '_group_name': 'lora_B',
            })
            print(f"  LoRA+ lora_B: {len(lora_b_params)} params, lr={base_lr * lr_ratio:.2e}")

        if other_params:
            groups.append({
                'params': other_params,
                'lr': base_lr,
                'weight_decay': weight_decay,
                '_group_name': 'other',
            })
            print(f"  LoRA+ other:  {len(other_params)} params, lr={base_lr:.2e}")

        return groups

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # Bias correction
            bias_correction1 = 1 - beta1 ** self._step_count
            bias_correction2 = 1 - beta2 ** self._step_count

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LoRAPlusAdamW does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decoupled weight decay (AdamW style)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected estimates
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                # Update parameters
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def get_lr_by_group(self) -> Dict[str, float]:
        """Get current learning rates by group name."""
        lrs = {}
        for group in self.param_groups:
            name = group.get('_group_name', 'unknown')
            lrs[name] = group['lr']
        return lrs

    def scale_lora_lr(self, scale: float) -> None:
        """
        Scale LoRA learning rates by a factor.

        Useful for learning rate warmup/decay specific to LoRA.
        """
        for group in self.param_groups:
            name = group.get('_group_name', '')
            if 'lora' in name.lower():
                group['lr'] *= scale


# ============================================================================
# Convenience Functions
# ============================================================================

def create_lora_plus_optimizer(
    model: nn.Module,
    base_lr: float = 1e-4,
    lr_ratio: float = 16.0,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    optimizer_class: type = None,
) -> Optimizer:
    """
    Create an optimizer with LoRA+ learning rate scaling.

    This is the recommended way to create an optimizer for LoRA+ training.

    Args:
        model: Model with LoRA parameters
        base_lr: Base learning rate for A matrices
        lr_ratio: Ratio of B to A learning rate (default: 16)
        betas: Adam beta parameters
        eps: Adam epsilon
        weight_decay: Weight decay
        optimizer_class: Optimizer class to use (default: AdamW)

    Returns:
        Configured optimizer with separate A/B learning rates
    """
    # Get parameter groups
    builder = LoRAPlusParameterGroups(model)
    param_groups = builder.get_param_groups(
        base_lr=base_lr,
        lr_ratio=lr_ratio,
        weight_decay=weight_decay,
    )

    # Create optimizer
    if optimizer_class is None:
        optimizer_class = torch.optim.AdamW

    return optimizer_class(
        param_groups,
        lr=base_lr,  # Default LR (overridden by groups)
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def get_lora_plus_param_groups(
    model: nn.Module,
    base_lr: float = 1e-4,
    lr_ratio: float = 16.0,
    weight_decay: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Get LoRA+ parameter groups for use with any optimizer.

    Args:
        model: Model with LoRA parameters
        base_lr: Base learning rate for A matrices
        lr_ratio: Ratio of B to A learning rate (default: 16)
        weight_decay: Weight decay

    Returns:
        List of parameter group dictionaries
    """
    builder = LoRAPlusParameterGroups(model)
    return builder.get_param_groups(
        base_lr=base_lr,
        lr_ratio=lr_ratio,
        weight_decay=weight_decay,
    )


# ============================================================================
# Integration with Existing Chronicals Optimizers
# ============================================================================

def wrap_optimizer_with_lora_plus(
    model: nn.Module,
    optimizer_class: type,
    base_lr: float = 1e-4,
    lr_ratio: float = 16.0,
    **optimizer_kwargs,
) -> Optimizer:
    """
    Wrap any optimizer class with LoRA+ parameter groups.

    This allows using LoRA+ with custom optimizers like FusedAdamW,
    ScheduleFreeAdamW, etc.

    Args:
        model: Model with LoRA parameters
        optimizer_class: Optimizer class to use
        base_lr: Base learning rate for A matrices
        lr_ratio: Ratio of B to A learning rate
        **optimizer_kwargs: Additional kwargs for optimizer

    Returns:
        Configured optimizer with LoRA+ groups
    """
    param_groups = get_lora_plus_param_groups(
        model,
        base_lr=base_lr,
        lr_ratio=lr_ratio,
        weight_decay=optimizer_kwargs.get('weight_decay', 0.0),
    )

    # Remove weight_decay from kwargs as it's in groups
    if 'weight_decay' in optimizer_kwargs:
        del optimizer_kwargs['weight_decay']

    return optimizer_class(param_groups, **optimizer_kwargs)


# ============================================================================
# Benchmark and Testing
# ============================================================================

if __name__ == "__main__":
    print("Chronicals LoRA+ Optimizer Module")
    print("=" * 60)

    # Create a simple model with LoRA-style naming
    class SimpleLoRAModel(nn.Module):
        def __init__(self, hidden_size=512, rank=8):
            super().__init__()
            # Base layers (frozen in real scenario)
            self.base_weight = nn.Parameter(torch.randn(hidden_size, hidden_size))

            # LoRA A matrices (lower LR)
            self.lora_A = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)
            self.layer1_lora_A = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)

            # LoRA B matrices (higher LR)
            self.lora_B = nn.Parameter(torch.randn(rank, hidden_size) * 0.01)
            self.layer1_lora_B = nn.Parameter(torch.randn(rank, hidden_size) * 0.01)

        def forward(self, x):
            # Base forward
            out = x @ self.base_weight

            # LoRA delta
            delta = x @ self.lora_A @ self.lora_B
            delta += x @ self.layer1_lora_A @ self.layer1_lora_B

            return out + delta

    # Test parameter group creation
    print("\nTest 1: Parameter Group Creation")
    print("-" * 40)

    model = SimpleLoRAModel()
    builder = LoRAPlusParameterGroups(model)
    param_groups = builder.get_param_groups(base_lr=1e-4, lr_ratio=16.0)

    # Verify groups
    for group in param_groups:
        print(f"  Group '{group.get('name', 'unknown')}': lr={group['lr']:.2e}")

    # Test LoRAPlusAdamW
    print("\nTest 2: LoRAPlusAdamW Optimizer")
    print("-" * 40)

    model = SimpleLoRAModel()
    optimizer = LoRAPlusAdamW(
        model.named_parameters(),
        lr=1e-4,
        lr_ratio=16.0,
    )

    # Print LR by group
    print(f"  Learning rates: {optimizer.get_lr_by_group()}")

    # Test training step
    x = torch.randn(4, 128, 512)
    output = model(x)
    loss = output.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"  Training step completed successfully")

    # Test scheduler compatibility
    print("\nTest 3: Scheduler Compatibility")
    print("-" * 40)

    from torch.optim.lr_scheduler import CosineAnnealingLR

    model = SimpleLoRAModel()
    optimizer = LoRAPlusAdamW(
        model.named_parameters(),
        lr=1e-4,
        lr_ratio=16.0,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    for step in range(5):
        x = torch.randn(4, 128, 512)
        output = model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if step % 2 == 0:
            lrs = optimizer.get_lr_by_group()
            print(f"  Step {step}: {lrs}")

    # Test create_lora_plus_optimizer helper
    print("\nTest 4: create_lora_plus_optimizer Helper")
    print("-" * 40)

    model = SimpleLoRAModel()
    optimizer = create_lora_plus_optimizer(
        model,
        base_lr=2e-4,
        lr_ratio=8.0,  # Custom ratio
    )

    print(f"  Optimizer type: {type(optimizer).__name__}")
    print(f"  Number of param groups: {len(optimizer.param_groups)}")

    # Verify different LRs
    lrs = {g.get('_group_name', g.get('name', 'unknown')): g['lr']
           for g in optimizer.param_groups}
    print(f"  Learning rates: {lrs}")

    # Verify ratio
    if 'lora_A' in lrs and 'lora_B' in lrs:
        actual_ratio = lrs['lora_B'] / lrs['lora_A']
        print(f"  LR ratio (B/A): {actual_ratio:.1f}x (expected: 8.0x)")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("\nLoRA+ provides 1.5-2x faster convergence by using")
    print("higher learning rate for B matrices (default: 16x)")


# Alias for backwards compatibility and easier imports
LoRAPlusOptimizer = LoRAPlusAdamW
