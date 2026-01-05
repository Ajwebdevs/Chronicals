"""
Chronicals Visual Reporter
===========================
Real-time visualization of training progress at every checkpoint.
Generates loss curves, throughput charts, and memory usage plots.

In Colab: Copy this entire cell, paste, and run to create visual_reporter.py
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    loss: float
    learning_rate: float
    tokens_per_sec: float
    memory_mb: float
    grad_norm: float = 0.0
    epoch: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class VisualReporter:
    """
    Visual reporting for training progress.

    Features:
    - Real-time loss curves
    - Throughput charts (tokens/sec)
    - Memory usage tracking
    - Checkpoint summaries
    - HTML report generation
    """

    def __init__(
        self,
        output_dir: str = "./reports",
        report_every: int = 100,
        save_plots: bool = True,
    ):
        self.output_dir = output_dir
        self.report_every = report_every
        self.save_plots = save_plots

        os.makedirs(output_dir, exist_ok=True)

        # Metrics history
        self.metrics_history: List[TrainingMetrics] = []
        self.checkpoint_metrics: List[Dict[str, Any]] = []

        # Baseline comparison
        self.baseline_metrics: Dict[str, List[float]] = {
            'huggingface': [],
            'unsloth': [],
            'chronicals': [],
        }

    def log_step(self, metrics: TrainingMetrics):
        """Log metrics for a training step."""
        self.metrics_history.append(metrics)

        # Auto-report at intervals
        if len(self.metrics_history) % self.report_every == 0:
            self.generate_live_report()

    def log_checkpoint(
        self,
        step: int,
        model_name: str,
        metrics: Dict[str, float],
    ):
        """Log checkpoint metrics."""
        checkpoint_data = {
            'step': step,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.checkpoint_metrics.append(checkpoint_data)

        # Save checkpoint report
        self._save_checkpoint_report(checkpoint_data)

    def log_baseline(self, method: str, metrics: Dict[str, float]):
        """Log baseline comparison metrics."""
        if method in self.baseline_metrics:
            self.baseline_metrics[method].append(metrics)

    def generate_live_report(self):
        """Generate live training report."""
        if not self.metrics_history:
            return

        try:
            import matplotlib.pyplot as plt
            self._plot_training_curves()
        except ImportError:
            print("matplotlib not available, skipping plots")

        # Print summary
        self._print_summary()

    def _plot_training_curves(self):
        """Plot training curves using matplotlib."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        steps = [m.step for m in self.metrics_history]
        losses = [m.loss for m in self.metrics_history]
        throughputs = [m.tokens_per_sec for m in self.metrics_history]
        memory = [m.memory_mb for m in self.metrics_history]
        lrs = [m.learning_rate for m in self.metrics_history]

        # Loss curve
        ax1 = axes[0, 0]
        ax1.plot(steps, losses, 'b-', linewidth=1.5, label='Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Smoothed loss
        if len(losses) > 10:
            window = min(50, len(losses) // 4)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax1.plot(steps[window-1:], smoothed, 'r-', linewidth=2, label='Smoothed')
            ax1.legend()

        # Throughput
        ax2 = axes[0, 1]
        ax2.plot(steps, throughputs, 'g-', linewidth=1.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Tokens/sec')
        ax2.set_title('Training Throughput')
        ax2.grid(True, alpha=0.3)

        # Add average line
        avg_throughput = np.mean(throughputs)
        ax2.axhline(y=avg_throughput, color='r', linestyle='--',
                    label=f'Avg: {avg_throughput:.0f}')
        ax2.legend()

        # Memory usage
        ax3 = axes[1, 0]
        ax3.plot(steps, memory, 'purple', linewidth=1.5)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_title('GPU Memory Usage')
        ax3.grid(True, alpha=0.3)

        # Learning rate
        ax4 = axes[1, 1]
        ax4.plot(steps, lrs, 'orange', linewidth=1.5)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(f'{self.output_dir}/training_curves.png', dpi=150)

        # Display in notebook
        try:
            from IPython.display import display, clear_output
            clear_output(wait=True)
            display(fig)
        except:
            plt.show()

        plt.close()

    def _print_summary(self):
        """Print training summary."""
        if not self.metrics_history:
            return

        recent = self.metrics_history[-100:]  # Last 100 steps

        avg_loss = np.mean([m.loss for m in recent])
        avg_throughput = np.mean([m.tokens_per_sec for m in recent])
        avg_memory = np.mean([m.memory_mb for m in recent])
        current_step = self.metrics_history[-1].step

        print("\n" + "=" * 60)
        print(f"Training Progress - Step {current_step}")
        print("=" * 60)
        print(f"  Loss (avg last 100):        {avg_loss:.4f}")
        print(f"  Throughput (avg):           {avg_throughput:.0f} tokens/sec")
        print(f"  Memory (avg):               {avg_memory:.0f} MB")
        print(f"  Total steps logged:         {len(self.metrics_history)}")
        print("=" * 60 + "\n")

    def _save_checkpoint_report(self, checkpoint_data: Dict):
        """Save checkpoint report."""
        filepath = f"{self.output_dir}/checkpoint_{checkpoint_data['step']}.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def generate_comparison_chart(self):
        """Generate comparison chart against baselines."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return

        if not any(self.baseline_metrics.values()):
            print("No baseline metrics to compare")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        methods = []
        throughputs = []
        memories = []
        losses = []

        for method, metrics_list in self.baseline_metrics.items():
            if metrics_list:
                methods.append(method)
                # Average metrics
                throughputs.append(np.mean([m.get('throughput', 0) for m in metrics_list]))
                memories.append(np.mean([m.get('memory_mb', 0) for m in metrics_list]))
                losses.append(np.mean([m.get('loss', 0) for m in metrics_list]))

        if not methods:
            return

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        # Throughput comparison
        ax1 = axes[0]
        bars1 = ax1.bar(methods, throughputs, color=colors[:len(methods)])
        ax1.set_ylabel('Tokens/sec')
        ax1.set_title('Throughput Comparison')
        ax1.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars1, throughputs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{val:.0f}', ha='center', va='bottom')

        # Memory comparison
        ax2 = axes[1]
        bars2 = ax2.bar(methods, memories, color=colors[:len(methods)])
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Peak Memory Comparison')
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars2, memories):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{val:.0f}', ha='center', va='bottom')

        # Loss comparison
        ax3 = axes[2]
        bars3 = ax3.bar(methods, losses, color=colors[:len(methods)])
        ax3.set_ylabel('Final Loss')
        ax3.set_title('Loss Comparison')
        ax3.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars3, losses):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom')

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(f'{self.output_dir}/comparison_chart.png', dpi=150)

        plt.show()
        plt.close()

    def generate_html_report(self) -> str:
        """Generate HTML report."""
        if not self.metrics_history:
            return "<html><body><p>No metrics to report</p></body></html>"

        recent = self.metrics_history[-100:]
        avg_loss = np.mean([m.loss for m in recent])
        avg_throughput = np.mean([m.tokens_per_sec for m in recent])
        avg_memory = np.mean([m.memory_mb for m in recent])

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chronicals Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 8px; }}
        .metric h3 {{ margin: 0 0 10px 0; color: #333; }}
        .metric .value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #2E86AB; color: white; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Chronicals Training Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="metrics-grid" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
        <div class="metric">
            <h3>Average Loss</h3>
            <div class="value">{avg_loss:.4f}</div>
        </div>
        <div class="metric">
            <h3>Throughput</h3>
            <div class="value">{avg_throughput:.0f} tok/s</div>
        </div>
        <div class="metric">
            <h3>Memory</h3>
            <div class="value">{avg_memory:.0f} MB</div>
        </div>
    </div>

    <h2>Training History</h2>
    <table>
        <tr>
            <th>Step</th>
            <th>Loss</th>
            <th>Throughput</th>
            <th>Memory (MB)</th>
            <th>LR</th>
        </tr>
        {''.join(f"<tr><td>{m.step}</td><td>{m.loss:.4f}</td><td>{m.tokens_per_sec:.0f}</td><td>{m.memory_mb:.0f}</td><td>{m.learning_rate:.2e}</td></tr>" for m in self.metrics_history[-20:])}
    </table>

    <div class="chart">
        <img src="training_curves.png" alt="Training Curves" style="max-width: 100%;">
    </div>

    <h2>Checkpoint History</h2>
    <table>
        <tr>
            <th>Step</th>
            <th>Model</th>
            <th>Timestamp</th>
        </tr>
        {''.join(f"<tr><td>{c['step']}</td><td>{c['model']}</td><td>{c['timestamp']}</td></tr>" for c in self.checkpoint_metrics)}
    </table>
</body>
</html>
"""

        # Save report
        filepath = f"{self.output_dir}/report.html"
        with open(filepath, 'w') as f:
            f.write(html)

        print(f"HTML report saved to: {filepath}")
        return html

    def save_metrics_json(self):
        """Save all metrics to JSON."""
        data = {
            'metrics_history': [
                {
                    'step': m.step,
                    'loss': m.loss,
                    'learning_rate': m.learning_rate,
                    'tokens_per_sec': m.tokens_per_sec,
                    'memory_mb': m.memory_mb,
                    'grad_norm': m.grad_norm,
                    'epoch': m.epoch,
                    'timestamp': m.timestamp,
                }
                for m in self.metrics_history
            ],
            'checkpoints': self.checkpoint_metrics,
            'baselines': self.baseline_metrics,
        }

        filepath = f"{self.output_dir}/metrics.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Metrics saved to: {filepath}")


def create_progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create ASCII progress bar."""
    progress = current / total
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percent = progress * 100
    return f'|{bar}| {percent:.1f}% ({current}/{total})'


if __name__ == "__main__":
    print("Testing Visual Reporter")
    print("=" * 50)

    # Create reporter
    reporter = VisualReporter(output_dir="./test_reports", save_plots=False)

    # Simulate training
    import random

    for step in range(1, 501):
        metrics = TrainingMetrics(
            step=step,
            loss=2.0 - step * 0.003 + random.uniform(-0.1, 0.1),
            learning_rate=1e-4 * (1 - step / 500),
            tokens_per_sec=50000 + random.uniform(-5000, 5000),
            memory_mb=15000 + random.uniform(-500, 500),
            grad_norm=random.uniform(0.5, 2.0),
        )
        reporter.log_step(metrics)

        if step % 100 == 0:
            reporter.log_checkpoint(step, "test-model", {'loss': metrics.loss})

    # Generate report
    reporter.generate_html_report()
    reporter.save_metrics_json()

    print("\nTest complete! Check ./test_reports/")
