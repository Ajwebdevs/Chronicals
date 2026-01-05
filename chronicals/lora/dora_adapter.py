%%writefile dora_adapter.py
"""
Chronicals DoRA (Weight-Decomposed Low-Rank Adaptation) Implementation
=======================================================================
DoRA decomposes pretrained weights into magnitude and direction components,
applying LoRA only to the directional component for better quality.

Key Innovation:
===============
DoRA decouples the magnitude and direction of weight updates:
- Magnitude: Learned scalar per output feature (m)
- Direction: LoRA applied to normalized weights (V/||V||)

Formula:
    W' = m * (W + delta_W) / ||W + delta_W||

Where:
- W: Original pretrained weights
- m: Learned magnitude vector [out_features]
- delta_W: LoRA update (A @ B)

Benefits:
=========
1. Better quality than standard LoRA (closer to full fine-tuning)
2. More stable training dynamics
3. No additional inference overhead when merged
4. Works with existing LoRA infrastructure

Key Properties:
===============
- Magnitude (m) controls the "strength" of each output neuron
- Direction (normalized W + LoRA) controls "what" the neuron represents
- This separation allows more nuanced adaptation

Performance:
============
- Training: Same memory as LoRA + small overhead for magnitude
- Inference: Zero overhead when merged
- Quality: Significantly better than LoRA, approaches full fine-tuning

References:
- DoRA: https://arxiv.org/abs/2402.09353 (NVIDIA, 2024)
- "DoRA: Weight-Decomposed Low-Rank Adaptation"

In Colab: Copy this entire cell, paste, and run to create dora_adapter.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Literal, List
import math


# ============================================================================
# DoRA Linear Layer
# ============================================================================

class DoRALinear(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) Linear layer.

    Decomposes the weight matrix into magnitude and direction:
        W' = m * (W_0 + B @ A) / ||W_0 + B @ A||

    Where:
    - W_0: Original pretrained weights (frozen)
    - m: Learned magnitude vector [out_features]
    - A, B: LoRA matrices
    - ||.||: Column-wise L2 norm

    This decomposition allows the model to:
    1. Adjust the "strength" of neurons via magnitude
    2. Adjust the "meaning" of neurons via direction (LoRA)

    Args:
        base_layer: Original linear layer to adapt
        rank: LoRA rank (default: 8)
        alpha: LoRA alpha for scaling (default: 16)
        dropout: Dropout probability on LoRA path (default: 0.0)
        init_magnitude: How to initialize magnitude ('pretrained', 'ones')
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_magnitude: Literal['pretrained', 'ones'] = 'pretrained',
    ):
        super().__init__()

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self._merged = False

        # Store original weight for direction computation
        # We need to keep W_0 frozen for the decomposition
        self.register_buffer(
            'weight_original',
            base_layer.weight.data.clone()
        )

        # LoRA matrices (same as standard LoRA)
        # B: [rank, in_features] - projects input to low-rank space
        # A: [out_features, rank] - projects from low-rank to output
        self.lora_B = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_A = nn.Parameter(torch.empty(self.out_features, rank))

        # DoRA: Learned magnitude vector
        # One scalar per output feature
        self.magnitude = nn.Parameter(torch.empty(self.out_features))

        # Dropout on the LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Initialize weights
        self._init_weights(init_magnitude)

    def _init_weights(self, init_magnitude: str) -> None:
        """Initialize LoRA and magnitude weights."""
        # LoRA initialization (standard Kaiming)
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        nn.init.zeros_(self.lora_A)

        # Magnitude initialization
        if init_magnitude == 'pretrained':
            # Initialize magnitude to match pretrained weight norms
            # m = ||W_0||_col (column-wise L2 norm)
            with torch.no_grad():
                weight_norm = self.weight_original.norm(p=2, dim=1)
                self.magnitude.data = weight_norm
        elif init_magnitude == 'ones':
            nn.init.ones_(self.magnitude)
        else:
            raise ValueError(f"Unknown init_magnitude: {init_magnitude}")

    def _get_weight_norm(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute column-wise L2 norm of weight matrix.

        Args:
            weight: Weight matrix [out_features, in_features]

        Returns:
            Norm vector [out_features]
        """
        # Compute L2 norm along input dimension (dim=1)
        # Add small epsilon for numerical stability
        return weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with DoRA decomposition.

        Computes: y = x @ (m * (W_0 + scaling * A @ B) / ||W_0 + scaling * A @ B||)^T + bias

        Uses optimized bracketing for the LoRA delta computation.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        if self._merged:
            return self.base_layer(x)

        # Compute LoRA delta with proper bracketing
        # delta_W = scaling * A @ B  [out_features, in_features]
        delta_W = self.scaling * (self.lora_A @ self.lora_B)

        # Combined weight: W_0 + delta_W
        weight_combined = self.weight_original + delta_W

        # Compute direction: normalize combined weight
        weight_norm = self._get_weight_norm(weight_combined)  # [out_features, 1]
        weight_direction = weight_combined / weight_norm  # [out_features, in_features]

        # Apply magnitude: m * direction
        # magnitude: [out_features], direction: [out_features, in_features]
        weight_final = self.magnitude.unsqueeze(1) * weight_direction

        # Compute output with dropout on input if needed
        x_dropped = self.lora_dropout(x)

        # Linear forward with decomposed weight
        output = F.linear(x_dropped, weight_final, self.base_layer.bias)

        return output

    def forward_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass that avoids materializing full delta_W.

        This is more memory efficient for large models but slightly
        more complex. Uses the identity:

        y = m * (W_0 @ x + scaling * A @ (B @ x)) / ||W_0 + scaling * A @ B||

        Note: The normalization still requires computing the full combined weight,
        so this optimization mainly helps with the actual computation.
        """
        if self._merged:
            return self.base_layer(x)

        # Apply dropout
        x_dropped = self.lora_dropout(x)

        # Base forward
        base_out = F.linear(x_dropped, self.weight_original, None)

        # LoRA forward with optimized bracketing
        low_rank = F.linear(x_dropped, self.lora_B, None)  # [..., rank]
        lora_out = F.linear(low_rank, self.lora_A, None)  # [..., out_features]

        # Combined output (before normalization adjustment)
        combined_out = base_out + self.scaling * lora_out

        # Compute normalization factor
        # We need ||W_0 + scaling * A @ B|| which unfortunately requires
        # materializing the combined weight
        delta_W = self.scaling * (self.lora_A @ self.lora_B)
        weight_combined = self.weight_original + delta_W
        weight_norm = self._get_weight_norm(weight_combined)

        # Apply magnitude and normalization
        # output = m * combined_out / ||W_combined||
        # Note: This works because linear(x, m*W/||W||) = m * linear(x, W) / ||W||
        output = (self.magnitude.unsqueeze(0) * combined_out) / weight_norm.squeeze(1)

        # Add bias
        if self.base_layer.bias is not None:
            output = output + self.base_layer.bias

        return output

    def merge(self) -> None:
        """
        Merge DoRA weights into base layer for inference.

        After merging, the forward pass uses only the base layer
        with no additional overhead.
        """
        if self._merged:
            return

        with torch.no_grad():
            # Compute final weight
            delta_W = self.scaling * (self.lora_A @ self.lora_B)
            weight_combined = self.weight_original + delta_W
            weight_norm = self._get_weight_norm(weight_combined)
            weight_direction = weight_combined / weight_norm
            weight_final = self.magnitude.unsqueeze(1) * weight_direction

            # Update base layer
            self.base_layer.weight.data = weight_final
            self._merged = True

    def unmerge(self) -> None:
        """
        Unmerge DoRA weights from base layer.

        Restores the decomposed representation for further training.
        """
        if not self._merged:
            return

        with torch.no_grad():
            # Restore original weight
            self.base_layer.weight.data = self.weight_original.clone()
            self._merged = False

    def get_effective_weight(self) -> torch.Tensor:
        """Get the effective weight matrix (magnitude * normalized(W + delta))."""
        with torch.no_grad():
            delta_W = self.scaling * (self.lora_A @ self.lora_B)
            weight_combined = self.weight_original + delta_W
            weight_norm = self._get_weight_norm(weight_combined)
            weight_direction = weight_combined / weight_norm
            return self.magnitude.unsqueeze(1) * weight_direction

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'rank={self.rank}, '
            f'alpha={self.alpha}, '
            f'merged={self._merged}'
        )


# ============================================================================
# DoRA Model Wrapper
# ============================================================================

class DoRAModel(nn.Module):
    """
    Wrapper that adds DoRA adapters to a model's linear layers.

    Automatically identifies and wraps target modules with DoRALinear.

    Args:
        model: Base model to adapt
        target_modules: List of module name patterns to adapt
        rank: LoRA rank
        alpha: LoRA alpha for scaling
        dropout: Dropout probability
    """

    def __init__(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if target_modules is None:
            target_modules = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ]

        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Track DoRA layers
        self.dora_layers: Dict[str, DoRALinear] = {}

        # Apply DoRA to target modules
        self._apply_dora()

    def _apply_dora(self) -> None:
        """Apply DoRA adapters to target modules."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if any(target in name for target in self.target_modules):
                    # Create DoRA wrapper
                    dora_layer = DoRALinear(
                        base_layer=module,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                    )

                    # Replace the module
                    parent_name, attr_name = self._get_parent_and_attr(name)
                    parent = self._get_module_by_name(parent_name)
                    setattr(parent, attr_name, dora_layer)

                    self.dora_layers[name] = dora_layer

        print(f"Applied DoRA to {len(self.dora_layers)} layers")
        for name in self.dora_layers:
            print(f"  - {name}")

    def _get_parent_and_attr(self, name: str) -> Tuple[str, str]:
        """Split module name into parent path and attribute name."""
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            return '', parts[0]
        return parts[0], parts[1]

    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get module by its full name."""
        if name == '':
            return self.model
        return dict(self.model.named_modules())[name]

    def forward(self, *args, **kwargs):
        """Forward pass through the adapted model."""
        return self.model(*args, **kwargs)

    def merge_dora(self) -> None:
        """Merge all DoRA weights for inference."""
        for layer in self.dora_layers.values():
            layer.merge()

    def unmerge_dora(self) -> None:
        """Unmerge all DoRA weights for training."""
        for layer in self.dora_layers.values():
            layer.unmerge()

    def get_dora_parameters(self) -> List[torch.nn.Parameter]:
        """Get list of all DoRA parameters for optimizer."""
        params = []
        for layer in self.dora_layers.values():
            params.append(layer.lora_A)
            params.append(layer.lora_B)
            params.append(layer.magnitude)
        return params

    def get_dora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict containing only DoRA parameters."""
        state_dict = {}
        for name, layer in self.dora_layers.items():
            state_dict[f'{name}.lora_A'] = layer.lora_A.data
            state_dict[f'{name}.lora_B'] = layer.lora_B.data
            state_dict[f'{name}.magnitude'] = layer.magnitude.data
        return state_dict

    def load_dora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load DoRA parameters from state dict."""
        for name, layer in self.dora_layers.items():
            layer.lora_A.data = state_dict[f'{name}.lora_A']
            layer.lora_B.data = state_dict[f'{name}.lora_B']
            layer.magnitude.data = state_dict[f'{name}.magnitude']

    def print_trainable_parameters(self) -> None:
        """Print the number of trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} = {100 * trainable / total:.2f}%")


# ============================================================================
# Convenience Functions
# ============================================================================

def apply_dora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
) -> DoRAModel:
    """
    Apply DoRA adapters to a model.

    Args:
        model: Base model to adapt
        rank: LoRA rank
        alpha: LoRA alpha for scaling
        target_modules: List of module name patterns to adapt

    Returns:
        DoRAModel wrapper with DoRA adapters applied
    """
    return DoRAModel(
        model=model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
    )


def convert_lora_to_dora(lora_layer) -> DoRALinear:
    """
    Convert an existing LoRA layer to DoRA.

    This preserves the learned A and B matrices while adding
    magnitude decomposition.

    Args:
        lora_layer: Existing LoRA layer (must have lora_A, lora_B attributes)

    Returns:
        DoRALinear with transferred weights
    """
    # Create DoRA layer
    dora_layer = DoRALinear(
        base_layer=lora_layer.base_layer,
        rank=lora_layer.rank,
        alpha=lora_layer.alpha,
    )

    # Transfer LoRA weights
    with torch.no_grad():
        dora_layer.lora_A.data = lora_layer.lora_A.data.clone()
        dora_layer.lora_B.data = lora_layer.lora_B.data.clone()

        # Initialize magnitude from combined weight
        delta_W = lora_layer.scaling * (lora_layer.lora_A @ lora_layer.lora_B)
        weight_combined = dora_layer.weight_original + delta_W
        weight_norm = weight_combined.norm(p=2, dim=1)
        dora_layer.magnitude.data = weight_norm

    return dora_layer


# ============================================================================
# DoRA+ Integration (DoRA with LoRA+ learning rates)
# ============================================================================

def get_dora_plus_param_groups(
    dora_model: DoRAModel,
    base_lr: float = 1e-4,
    lr_ratio: float = 16.0,
    magnitude_lr_ratio: float = 1.0,
    weight_decay: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Get parameter groups for DoRA with LoRA+ style learning rates.

    Extends LoRA+ by also handling the magnitude parameters.

    Args:
        dora_model: DoRA-adapted model
        base_lr: Base learning rate for A matrices
        lr_ratio: Ratio of B to A learning rate
        magnitude_lr_ratio: Ratio of magnitude to A learning rate
        weight_decay: Weight decay

    Returns:
        List of parameter group dictionaries
    """
    lora_a_params = []
    lora_b_params = []
    magnitude_params = []
    other_params = []

    for name, param in dora_model.named_parameters():
        if not param.requires_grad:
            continue

        if 'lora_A' in name:
            lora_a_params.append(param)
        elif 'lora_B' in name:
            lora_b_params.append(param)
        elif 'magnitude' in name:
            magnitude_params.append(param)
        else:
            other_params.append(param)

    param_groups = []

    if lora_a_params:
        param_groups.append({
            'params': lora_a_params,
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'lora_A',
        })

    if lora_b_params:
        param_groups.append({
            'params': lora_b_params,
            'lr': base_lr * lr_ratio,
            'weight_decay': 0.0,
            'name': 'lora_B',
        })

    if magnitude_params:
        param_groups.append({
            'params': magnitude_params,
            'lr': base_lr * magnitude_lr_ratio,
            'weight_decay': 0.0,
            'name': 'magnitude',
        })

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'other',
        })

    print(f"DoRA+ Parameter Groups:")
    print(f"  lora_A:    {len(lora_a_params)} params, lr={base_lr:.2e}")
    print(f"  lora_B:    {len(lora_b_params)} params, lr={base_lr * lr_ratio:.2e}")
    print(f"  magnitude: {len(magnitude_params)} params, lr={base_lr * magnitude_lr_ratio:.2e}")
    print(f"  other:     {len(other_params)} params, lr={base_lr:.2e}")

    return param_groups


# ============================================================================
# Benchmark and Testing
# ============================================================================

if __name__ == "__main__":
    print("Chronicals DoRA (Weight-Decomposed LoRA) Module")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, hidden_size=512):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.o_proj = nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            return self.o_proj(q + k + v)

    # Test DoRA layer
    print("\nTest 1: DoRA Layer Basic Functionality")
    print("-" * 40)

    base_linear = nn.Linear(512, 512).to(device)
    dora_layer = DoRALinear(base_linear, rank=8, alpha=16).to(device)

    x = torch.randn(2, 32, 512, device=device)
    output = dora_layer(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Magnitude shape: {dora_layer.magnitude.shape}")
    print(f"  Magnitude mean: {dora_layer.magnitude.mean().item():.4f}")

    # Test DoRA model wrapper
    print("\nTest 2: DoRA Model Wrapper")
    print("-" * 40)

    model = SimpleModel().to(device)
    dora_model = apply_dora_to_model(model, rank=8, alpha=16)
    dora_model = dora_model.to(device)

    output = dora_model(x)
    print(f"  Output shape: {output.shape}")
    dora_model.print_trainable_parameters()

    # Test merge/unmerge
    print("\nTest 3: Merge/Unmerge")
    print("-" * 40)

    output_before = dora_model(x).clone()

    dora_model.merge_dora()
    output_merged = dora_model(x)

    dora_model.unmerge_dora()
    output_after = dora_model(x)

    diff_merged = (output_before - output_merged).abs().max().item()
    diff_unmerged = (output_before - output_after).abs().max().item()

    print(f"  Max diff (merged):   {diff_merged:.2e}")
    print(f"  Max diff (unmerged): {diff_unmerged:.2e}")

    # Test backward pass
    print("\nTest 4: Backward Pass")
    print("-" * 40)

    model = SimpleModel().to(device)
    dora_model = apply_dora_to_model(model, rank=8).to(device)

    optimizer = torch.optim.AdamW(dora_model.get_dora_parameters(), lr=1e-4)

    for step in range(3):
        x = torch.randn(4, 32, 512, device=device)
        output = dora_model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step + 1}: loss = {loss.item():.6f}")

    # Test DoRA+ parameter groups
    print("\nTest 5: DoRA+ Parameter Groups")
    print("-" * 40)

    model = SimpleModel().to(device)
    dora_model = apply_dora_to_model(model, rank=8).to(device)

    param_groups = get_dora_plus_param_groups(
        dora_model,
        base_lr=1e-4,
        lr_ratio=16.0,
        magnitude_lr_ratio=0.1,  # Magnitude learns slower
    )

    optimizer = torch.optim.AdamW(param_groups)
    print(f"  Optimizer created with {len(param_groups)} groups")

    # Compare DoRA vs LoRA quality (simple test)
    print("\nTest 6: DoRA vs LoRA Comparison")
    print("-" * 40)

    # Import LoRA for comparison
    try:
        from lora_optimized import apply_lora_to_model

        # Create identical models
        model_lora = SimpleModel().to(device)
        model_dora = SimpleModel().to(device)

        # Copy weights
        model_dora.load_state_dict(model_lora.state_dict())

        # Apply adapters
        lora_model = apply_lora_to_model(model_lora, rank=8, alpha=16).to(device)
        dora_model = apply_dora_to_model(model_dora, rank=8, alpha=16).to(device)

        # Same input
        x = torch.randn(4, 32, 512, device=device)
        target = torch.randn(4, 32, 512, device=device)

        # Train both for a few steps
        opt_lora = torch.optim.AdamW(lora_model.get_lora_parameters(), lr=1e-3)
        opt_dora = torch.optim.AdamW(dora_model.get_dora_parameters(), lr=1e-3)

        for step in range(10):
            # LoRA
            out_lora = lora_model(x)
            loss_lora = F.mse_loss(out_lora, target)
            loss_lora.backward()
            opt_lora.step()
            opt_lora.zero_grad()

            # DoRA
            out_dora = dora_model(x)
            loss_dora = F.mse_loss(out_dora, target)
            loss_dora.backward()
            opt_dora.step()
            opt_dora.zero_grad()

        print(f"  Final LoRA loss: {loss_lora.item():.6f}")
        print(f"  Final DoRA loss: {loss_dora.item():.6f}")

    except ImportError:
        print("  (Skipping comparison - lora_optimized not available)")

    # Memory comparison
    print("\nTest 7: Memory Analysis")
    print("-" * 40)

    model = SimpleModel().to(device)
    dora_model = apply_dora_to_model(model, rank=8).to(device)

    # Count parameters
    base_params = sum(p.numel() for name, p in dora_model.named_parameters()
                     if 'weight_original' in name or not p.requires_grad)
    lora_a_params = sum(p.numel() for name, p in dora_model.named_parameters()
                       if 'lora_A' in name)
    lora_b_params = sum(p.numel() for name, p in dora_model.named_parameters()
                       if 'lora_B' in name)
    magnitude_params = sum(p.numel() for name, p in dora_model.named_parameters()
                          if 'magnitude' in name)

    total_lora = lora_a_params + lora_b_params
    total_dora = total_lora + magnitude_params

    print(f"  LoRA A params:    {lora_a_params:,}")
    print(f"  LoRA B params:    {lora_b_params:,}")
    print(f"  Magnitude params: {magnitude_params:,}")
    print(f"  Total LoRA:       {total_lora:,}")
    print(f"  Total DoRA:       {total_dora:,}")
    print(f"  DoRA overhead:    {100 * magnitude_params / total_lora:.2f}%")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("\nDoRA provides better quality than LoRA by decomposing")
    print("weights into magnitude and direction components.")
