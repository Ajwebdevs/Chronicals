%%writefile optimizers.py
"""
Chronicals Advanced Optimizers
===============================
Production-grade optimizers based on latest 2025 research.
- Schedule-Free AdamW (Meta AI)
- Muon optimizer (hidden layers only)
- 8-bit Adam (bitsandbytes)
- Adam-atan2 (DeepSeek/HRM style)

In Colab: Copy this entire cell, paste, and run to create optimizers.py
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, List, Optional, Tuple, Callable, Iterable
import math


# =============================================================================
# Schedule-Free AdamW (Meta AI - MLCommons 2024 Winner)
# =============================================================================

class ScheduleFreeAdamW(Optimizer):
    """
    Schedule-Free AdamW optimizer from Meta AI.

    Eliminates the need for learning rate schedules by using
    interpolation and averaging instead of momentum.

    Key benefits:
    - No need to specify total training steps T
    - Same memory as AdamW
    - Won MLCommons 2024 AlgoPerf Challenge

    Reference: https://github.com/facebookresearch/schedule_free
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,  # EMA coefficient for z sequence
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            warmup_steps=warmup_steps, r=r,
        )
        super().__init__(params, defaults)

        self._step = 0

    def _get_lr_multiplier(self) -> float:
        """Get warmup multiplier."""
        warmup_steps = self.defaults['warmup_steps']
        if warmup_steps == 0 or self._step >= warmup_steps:
            return 1.0
        return self._step / warmup_steps

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1
        lr_mult = self._get_lr_multiplier()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr'] * lr_mult
            eps = group['eps']
            weight_decay = group['weight_decay']
            r = group['r']

            # Bias correction
            bias_correction1 = 1 - beta1 ** self._step
            bias_correction2 = 1 - beta2 ** self._step

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ScheduleFreeAdamW does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['z'] = p.clone()  # Averaging sequence

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                z = state['z']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Compute step size
                step_size = lr / bias_correction1

                # Weight decay
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

                # Update z (averaging sequence)
                z.addcdiv_(exp_avg, denom, value=-step_size)

                # Interpolate to get x
                # x = (1 - 1/t) * x + (1/t) * z
                c = 1.0 / self._step
                p.mul_(1 - c).add_(z, alpha=c)

        return loss

    def eval_mode(self):
        """Switch to evaluation mode - use z instead of x."""
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'z' in self.state[p]:
                    # Store x, use z
                    self.state[p]['x_backup'] = p.data.clone()
                    p.data.copy_(self.state[p]['z'])

    def train_mode(self):
        """Switch back to training mode."""
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'x_backup' in self.state[p]:
                    p.data.copy_(self.state[p]['x_backup'])
                    del self.state[p]['x_backup']


# =============================================================================
# Muon Optimizer (2025 - Spectral Normalization)
# =============================================================================

class Muon(Optimizer):
    """
    Muon optimizer using polar decomposition for spectrally normalized updates.

    Key features:
    - Uses Newton-Schulz iteration for efficient orthogonalization
    - Only maintains first moment (half the memory of Adam)
    - ~2x compute efficiency vs AdamW in scaling laws

    IMPORTANT: Only use for hidden layer weights (not embeddings, classifiers, biases)

    Reference: https://github.com/KellerJordan/Muon
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,  # Newton-Schulz iterations
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Apply Newton-Schulz iteration to orthogonalize matrix G.

        Converges to the polar factor (orthogonal part) of G.
        """
        # Ensure G is 2D
        original_shape = G.shape
        if G.dim() != 2:
            G = G.view(G.size(0), -1)

        # Scale for numerical stability
        scale = (G.shape[0] * G.shape[1]) ** 0.5 / G.norm()
        X = G * scale

        # Newton-Schulz iterations
        for _ in range(steps):
            A = X @ X.T
            B = A @ X
            # X = 1.5 * X - 0.5 * B
            X = X.mul_(1.5).sub_(B, alpha=0.5)

        return X.view(original_shape) / scale

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Only apply to 2D+ tensors (skip biases, 1D params)
                if p.dim() < 2:
                    # Fallback to SGD with momentum for 1D params
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                    p.add_(grad, alpha=-lr)
                    continue

                state = self.state[p]

                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf.clone()

                # Orthogonalize the update using Newton-Schulz
                update_orth = self.newton_schulz_orthogonalize(update, steps=ns_steps)

                # Apply update
                p.add_(update_orth, alpha=-lr)

        return loss


# =============================================================================
# 8-bit Adam (bitsandbytes wrapper)
# =============================================================================

class Adam8bit(Optimizer):
    """
    8-bit Adam optimizer wrapper.

    Uses bitsandbytes for memory-efficient optimizer states.
    Falls back to regular AdamW if bitsandbytes is not available.

    Memory savings: ~75% reduction in optimizer state memory
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        try:
            import bitsandbytes as bnb
            self._optimizer = bnb.optim.Adam8bit(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
            self._is_8bit = True
            print("  ✓ 8-bit Adam enabled (bitsandbytes)")
        except ImportError:
            print("  ○ bitsandbytes not available, using standard AdamW")
            self._optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
            self._is_8bit = False

        # Copy attributes for compatibility
        self.param_groups = self._optimizer.param_groups
        self.state = self._optimizer.state
        self.defaults = self._optimizer.defaults

    def step(self, closure: Optional[Callable] = None):
        return self._optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        return self._optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self._optimizer.load_state_dict(state_dict)


# =============================================================================
# Adam-atan2 (DeepSeek/HRM Style)
# =============================================================================

class AdamAtan2(Optimizer):
    """
    Adam-atan2 optimizer using atan2 for update computation.

    Key innovation: Uses atan2(m, sqrt(v)) instead of m / (sqrt(v) + eps)
    - Bounded updates (values in [-pi, pi])
    - No epsilon hyperparameter needed
    - Improved numerical stability
    - Better plasticity for continual learning

    Reference: Used by DeepSeek, HRM models
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        a: float = 1.27,  # Scaling factor for atan2
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a)
        super().__init__(params, defaults)
        self._step = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            a = group['a']

            # Bias correction
            bias_correction1 = 1 - beta1 ** self._step
            bias_correction2 = 1 - beta2 ** self._step

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected estimates
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # atan2-based update
                # update = a * atan2(m, sqrt(v))
                update = a * torch.atan2(m_hat, v_hat.sqrt())

                # Weight decay (decoupled)
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss


# =============================================================================
# Hybrid Optimizer (Muon for hidden layers + AdamW for rest)
# =============================================================================

class HybridMuonAdamW:
    """
    Hybrid optimizer that uses Muon for hidden layers and AdamW for the rest.

    Based on Muon paper recommendations:
    - Muon for transformer block weights (better scaling)
    - AdamW for embeddings, classifiers, biases, LayerNorm
    """

    def __init__(
        self,
        model: nn.Module,
        lr_muon: float = 0.02,
        lr_adamw: float = 1e-4,
        momentum: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        muon_layers: Optional[List[str]] = None,
    ):
        """
        Args:
            model: The model to optimize
            lr_muon: Learning rate for Muon parameters
            lr_adamw: Learning rate for AdamW parameters
            momentum: Muon momentum
            betas: AdamW betas
            weight_decay: Weight decay for both
            muon_layers: List of layer name patterns to use Muon on
                        Default: ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                 'gate_proj', 'up_proj', 'down_proj',
                                 'attention', 'mlp']
        """
        if muon_layers is None:
            muon_layers = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
                'attention', 'mlp', 'self_attn', 'feed_forward'
            ]

        # Separate parameters
        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this is a Muon-eligible layer
            is_muon = any(layer in name for layer in muon_layers)
            is_2d = param.dim() >= 2

            if is_muon and is_2d:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        print(f"  Hybrid optimizer: {len(muon_params)} Muon params, {len(adamw_params)} AdamW params")

        # Create optimizers
        self.muon = Muon(muon_params, lr=lr_muon, momentum=momentum) if muon_params else None
        self.adamw = torch.optim.AdamW(
            adamw_params, lr=lr_adamw, betas=betas, weight_decay=weight_decay
        ) if adamw_params else None

        # Combined param_groups for compatibility
        self.param_groups = []
        if self.muon:
            self.param_groups.extend(self.muon.param_groups)
        if self.adamw:
            self.param_groups.extend(self.adamw.param_groups)

    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step."""
        loss = None
        if self.muon:
            loss = self.muon.step(closure if self.adamw is None else None)
        if self.adamw:
            loss = self.adamw.step(closure)
        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        if self.muon:
            self.muon.zero_grad(set_to_none)
        if self.adamw:
            self.adamw.zero_grad(set_to_none)

    def state_dict(self):
        """Return combined state dict."""
        return {
            'muon': self.muon.state_dict() if self.muon else None,
            'adamw': self.adamw.state_dict() if self.adamw else None,
        }

    def load_state_dict(self, state_dict):
        """Load combined state dict."""
        if self.muon and state_dict.get('muon'):
            self.muon.load_state_dict(state_dict['muon'])
        if self.adamw and state_dict.get('adamw'):
            self.adamw.load_state_dict(state_dict['adamw'])


# =============================================================================
# Learning Rate Schedules
# =============================================================================

class WSDScheduler:
    """
    Warmup-Stable-Decay (WSD) learning rate schedule.

    Used by DeepSeek-V3, ERNIE 4.5, and other SOTA models.

    Three phases:
    1. Warmup: Linear ramp-up
    2. Stable: Constant peak LR (majority of training)
    3. Decay: Linear or cosine decay to min_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_ratio: float = 0.05,
        stable_ratio: float = 0.80,
        min_lr_ratio: float = 0.1,
        decay_type: str = "cosine",  # "linear" or "cosine"
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.stable_steps = int(total_steps * stable_ratio)
        self.decay_steps = total_steps - self.warmup_steps - self.stable_steps
        self.min_lr_ratio = min_lr_ratio
        self.decay_type = decay_type

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._step = 0

    def get_lr(self) -> List[float]:
        """Get current learning rate."""
        step = self._step

        if step < self.warmup_steps:
            # Warmup phase: linear ramp
            mult = step / self.warmup_steps
        elif step < self.warmup_steps + self.stable_steps:
            # Stable phase: constant
            mult = 1.0
        else:
            # Decay phase
            decay_step = step - self.warmup_steps - self.stable_steps
            decay_progress = decay_step / max(self.decay_steps, 1)

            if self.decay_type == "cosine":
                mult = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                    1 + math.cos(math.pi * decay_progress)
                )
            else:  # linear
                mult = 1.0 - (1 - self.min_lr_ratio) * decay_progress

        return [base_lr * mult for base_lr in self.base_lrs]

    def step(self):
        """Update learning rate."""
        self._step += 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def get_last_lr(self) -> List[float]:
        """Get last computed learning rate."""
        return self.get_lr()

    def state_dict(self):
        return {'step': self._step}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']


class OneCycleLR:
    """
    One-Cycle learning rate schedule.

    Single cycle: ramp up -> ramp down -> annihilate
    Up to 10x faster convergence with better generalization.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,  # Fraction of cycle for ramp-up
        anneal_strategy: str = "cos",  # "cos" or "linear"
        div_factor: float = 25.0,  # initial_lr = max_lr / div_factor
        final_div_factor: float = 1e4,  # min_lr = initial_lr / final_div_factor
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor

        self._step = 0

    def get_lr(self) -> float:
        """Get current learning rate."""
        step = self._step

        if step < self.total_steps * self.pct_start:
            # Ramp up phase
            progress = step / (self.total_steps * self.pct_start)
            if self.anneal_strategy == "cos":
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (
                    1 - math.cos(math.pi * progress)
                ) / 2
            else:
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Ramp down phase
            progress = (step - self.total_steps * self.pct_start) / (
                self.total_steps * (1 - self.pct_start)
            )
            if self.anneal_strategy == "cos":
                lr = self.min_lr + (self.max_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                ) / 2
            else:
                lr = self.max_lr - (self.max_lr - self.min_lr) * progress

        return lr

    def step(self):
        """Update learning rate."""
        self._step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self) -> List[float]:
        return [self.get_lr()]

    def state_dict(self):
        return {'step': self._step}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']


# =============================================================================
# Utility Functions
# =============================================================================

def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    **kwargs,
) -> Optimizer:
    """
    Create optimizer based on type.

    Args:
        model: Model to optimize
        optimizer_type: One of "adamw", "fused_adamw", "schedule_free",
                       "muon", "hybrid_muon", "8bit_adam", "adam_atan2"
        lr: Learning rate
        weight_decay: Weight decay
        betas: Adam betas
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured optimizer
    """
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    elif optimizer_type == "fused_adamw":
        try:
            from fused_adamw import FusedAdamW
            return FusedAdamW(
                params, lr=lr, betas=betas, weight_decay=weight_decay,
                max_grad_norm=kwargs.get('max_grad_norm', 1.0)
            )
        except ImportError:
            print("FusedAdamW not available, falling back to AdamW")
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    elif optimizer_type == "schedule_free":
        return ScheduleFreeAdamW(
            params, lr=lr, betas=betas, weight_decay=weight_decay,
            warmup_steps=kwargs.get('warmup_steps', 0)
        )

    elif optimizer_type == "muon":
        return Muon(params, lr=lr, momentum=kwargs.get('momentum', 0.95))

    elif optimizer_type == "hybrid_muon":
        return HybridMuonAdamW(
            model, lr_muon=kwargs.get('lr_muon', 0.02), lr_adamw=lr,
            weight_decay=weight_decay, betas=betas
        )

    elif optimizer_type == "8bit_adam":
        return Adam8bit(params, lr=lr, betas=betas, weight_decay=weight_decay)

    elif optimizer_type == "adam_atan2":
        return AdamAtan2(params, lr=lr, betas=betas, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 1000,
    warmup_ratio: float = 0.03,
    **kwargs,
):
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_type: One of "cosine", "wsd", "one_cycle", "constant"
        total_steps: Total training steps
        warmup_ratio: Warmup ratio
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Configured scheduler
    """
    warmup_steps = int(total_steps * warmup_ratio)

    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps,
            eta_min=kwargs.get('min_lr', 0)
        )
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    elif scheduler_type == "wsd":
        return WSDScheduler(
            optimizer, total_steps=total_steps, warmup_ratio=warmup_ratio,
            stable_ratio=kwargs.get('stable_ratio', 0.80),
            min_lr_ratio=kwargs.get('min_lr_ratio', 0.1),
            decay_type=kwargs.get('decay_type', 'cosine')
        )

    elif scheduler_type == "one_cycle":
        max_lr = optimizer.param_groups[0]['lr']
        return OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=warmup_ratio
        )

    elif scheduler_type == "constant":
        from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR

        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        constant = ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps)
        return SequentialLR(optimizer, schedulers=[warmup, constant], milestones=[warmup_steps])

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    print("Chronicals Optimizers Module")
    print("=" * 50)
    print("\nAvailable optimizers:")
    print("  - ScheduleFreeAdamW (Meta AI)")
    print("  - Muon (spectral normalization)")
    print("  - HybridMuonAdamW (Muon + AdamW)")
    print("  - Adam8bit (bitsandbytes)")
    print("  - AdamAtan2 (DeepSeek/HRM)")
    print("\nAvailable schedulers:")
    print("  - WSDScheduler (Warmup-Stable-Decay)")
    print("  - OneCycleLR")
    print("  - Cosine with warmup")
    print("  - Constant with warmup")
