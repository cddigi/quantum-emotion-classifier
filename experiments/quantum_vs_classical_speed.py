#!/usr/bin/env python3
"""
Quantum vs Classical Speed Comparison Experiment

Compares training and inference speeds between quantum and classical approaches
for emotion classification across different problem sizes.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantum_emotion.classifier import QuantumTextClassifier
from quantum_emotion.hybrid import HybridTrainer
from quantum_emotion.utils import create_emotion_dataset, split_dataset, create_data_loaders

class ClassicalEmotionClassifier(nn.Module):
    """Classical neural network for emotion classification comparison."""

    def __init__(self, feature_dim=4, n_classes=4, hidden_dims=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]  # Comparable to quantum parameter count

        layers = []
        prev_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

        # Store config for comparison
        self.config = {
            'feature_dim': feature_dim,
            'n_classes': n_classes,
            'hidden_dims': hidden_dims,
            'total_params': sum(p.numel() for p in self.parameters())
        }

    def forward(self, x):
        return self.network(x)

class ClassicalTrainer:
    """Classical trainer to match HybridTrainer interface."""

    def __init__(self, model, learning_rate=0.01, device=None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader=None, epochs=10, verbose=False):
        """Training loop."""
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)

            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")

        return self.history

def benchmark_model(model_type, n_qubits_or_params, n_samples=100, epochs=3, device='cpu'):
    """Benchmark a single model configuration."""

    try:
        # Create dataset
        features, labels, _ = create_emotion_dataset(n_samples=n_samples)
        splits = split_dataset(features, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42)
        loaders = create_data_loaders(splits, batch_size=16)

        if model_type == 'quantum':
            # Quantum model
            model = QuantumTextClassifier(
                n_qubits=n_qubits_or_params,
                n_classes=4,
                n_layers=1,
                feature_dim=features.shape[1]
            )
            trainer = HybridTrainer(model, learning_rate=0.01, device=torch.device(device))

        else:
            # Classical model with comparable parameter count
            quantum_params = n_qubits_or_params * 3 * 1  # qubits * 3 rotations * 1 layer
            hidden_dim = max(16, quantum_params // 2)  # Approximate comparable complexity

            model = ClassicalEmotionClassifier(
                feature_dim=features.shape[1],
                n_classes=4,
                hidden_dims=[hidden_dim, hidden_dim//2]
            )
            trainer = ClassicalTrainer(model, learning_rate=0.01, device=torch.device(device))

        # Benchmark training time
        start_time = time.time()
        history = trainer.train(
            loaders['train'],
            loaders['val'],
            epochs=epochs,
            verbose=False
        )
        training_time = time.time() - start_time

        # Benchmark inference time
        test_data = next(iter(loaders['test']))[0]

        # Warm-up run
        with torch.no_grad():
            _ = model(test_data)

        # Timed inference runs
        inference_times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = model(test_data)
            inference_times.append(time.time() - start_time)

        avg_inference_time = np.mean(inference_times)

        # Get final accuracy
        final_accuracy = history['val_accuracy'][-1] if history['val_accuracy'] else 0.0

        # Get parameter count
        if model_type == 'quantum':
            param_count = model.quantum_params.numel() + sum(p.numel() for p in model.classical_preprocess.parameters()) + sum(p.numel() for p in model.classical_postprocess.parameters())
        else:
            param_count = model.config['total_params']

        return {
            'model_type': model_type,
            'n_qubits_or_size': n_qubits_or_params,
            'training_time': training_time,
            'inference_time': avg_inference_time,
            'accuracy': final_accuracy,
            'param_count': param_count,
            'samples_per_second_train': n_samples * epochs / training_time,
            'samples_per_second_inference': len(test_data) / avg_inference_time
        }

    except Exception as e:
        print(f"Error benchmarking {model_type} with {n_qubits_or_params}: {e}")
        return None

def run_speed_comparison(max_qubits=8, n_samples=100, epochs=3):
    """Run comprehensive speed comparison."""
    print("🏁 Quantum vs Classical Speed Comparison")
    print("=" * 50)

    results = []

    # Test different model sizes
    for size in range(2, max_qubits + 1):
        print(f"\n📊 Testing size {size}...")

        # Quantum benchmark
        print(f"  ⚛️  Quantum ({size} qubits)...", end=" ")
        quantum_result = benchmark_model('quantum', size, n_samples, epochs, 'cpu')
        if quantum_result:
            results.append(quantum_result)
            print(f"✓ {quantum_result['training_time']:.2f}s")
        else:
            print("✗ Failed")

        # Classical benchmark
        print(f"  🧠 Classical (comparable params)...", end=" ")
        classical_result = benchmark_model('classical', size, n_samples, epochs, 'cpu')
        if classical_result:
            results.append(classical_result)
            print(f"✓ {classical_result['training_time']:.2f}s")
        else:
            print("✗ Failed")

    return pd.DataFrame(results)

def create_speed_comparison_charts(df, save_dir):
    """Create comprehensive speed comparison charts."""

    # Separate quantum and classical results
    quantum_df = df[df['model_type'] == 'quantum'].copy()
    classical_df = df[df['model_type'] == 'classical'].copy()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quantum vs Classical Performance Comparison', fontsize=16, fontweight='bold')

    # Colors
    quantum_color = '#2E86C1'  # Blue
    classical_color = '#E74C3C'  # Red

    # 1. Training Time Comparison
    if not quantum_df.empty and not classical_df.empty:
        axes[0, 0].plot(quantum_df['n_qubits_or_size'], quantum_df['training_time'],
                       'o-', linewidth=2, markersize=8, color=quantum_color, label='Quantum')
        axes[0, 0].plot(classical_df['n_qubits_or_size'], classical_df['training_time'],
                       's-', linewidth=2, markersize=8, color=classical_color, label='Classical')
    axes[0, 0].set_xlabel('Model Size')
    axes[0, 0].set_ylabel('Training Time (seconds)')
    axes[0, 0].set_title('Training Speed Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Inference Time Comparison
    if not quantum_df.empty and not classical_df.empty:
        axes[0, 1].plot(quantum_df['n_qubits_or_size'], quantum_df['inference_time'] * 1000,
                       'o-', linewidth=2, markersize=8, color=quantum_color, label='Quantum')
        axes[0, 1].plot(classical_df['n_qubits_or_size'], classical_df['inference_time'] * 1000,
                       's-', linewidth=2, markersize=8, color=classical_color, label='Classical')
    axes[0, 1].set_xlabel('Model Size')
    axes[0, 1].set_ylabel('Inference Time (milliseconds)')
    axes[0, 1].set_title('Inference Speed Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Accuracy Comparison
    if not quantum_df.empty and not classical_df.empty:
        axes[0, 2].plot(quantum_df['n_qubits_or_size'], quantum_df['accuracy'] * 100,
                       'o-', linewidth=2, markersize=8, color=quantum_color, label='Quantum')
        axes[0, 2].plot(classical_df['n_qubits_or_size'], classical_df['accuracy'] * 100,
                       's-', linewidth=2, markersize=8, color=classical_color, label='Classical')
    axes[0, 2].set_xlabel('Model Size')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Accuracy Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Parameter Count Comparison
    if not quantum_df.empty and not classical_df.empty:
        axes[1, 0].plot(quantum_df['n_qubits_or_size'], quantum_df['param_count'],
                       'o-', linewidth=2, markersize=8, color=quantum_color, label='Quantum')
        axes[1, 0].plot(classical_df['n_qubits_or_size'], classical_df['param_count'],
                       's-', linewidth=2, markersize=8, color=classical_color, label='Classical')
    axes[1, 0].set_xlabel('Model Size')
    axes[1, 0].set_ylabel('Parameter Count')
    axes[1, 0].set_title('Model Complexity Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Training Throughput
    if not quantum_df.empty and not classical_df.empty:
        axes[1, 1].plot(quantum_df['n_qubits_or_size'], quantum_df['samples_per_second_train'],
                       'o-', linewidth=2, markersize=8, color=quantum_color, label='Quantum')
        axes[1, 1].plot(classical_df['n_qubits_or_size'], classical_df['samples_per_second_train'],
                       's-', linewidth=2, markersize=8, color=classical_color, label='Classical')
    axes[1, 1].set_xlabel('Model Size')
    axes[1, 1].set_ylabel('Samples/Second (Training)')
    axes[1, 1].set_title('Training Throughput')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Speed Ratio (Classical/Quantum)
    if not quantum_df.empty and not classical_df.empty:
        # Calculate speed ratios
        merged_df = pd.merge(quantum_df, classical_df, on='n_qubits_or_size', suffixes=('_quantum', '_classical'))
        training_ratio = merged_df['training_time_classical'] / merged_df['training_time_quantum']
        inference_ratio = merged_df['inference_time_classical'] / merged_df['inference_time_quantum']

        axes[1, 2].plot(merged_df['n_qubits_or_size'], training_ratio,
                       'o-', linewidth=2, markersize=8, color='green', label='Training Speed Ratio')
        axes[1, 2].plot(merged_df['n_qubits_or_size'], inference_ratio,
                       's-', linewidth=2, markersize=8, color='purple', label='Inference Speed Ratio')
        axes[1, 2].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')

    axes[1, 2].set_xlabel('Model Size')
    axes[1, 2].set_ylabel('Speed Ratio (Classical/Quantum)')
    axes[1, 2].set_title('Relative Performance\n(>1 = Classical Faster)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = save_dir / 'quantum_vs_classical_speed.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Speed comparison charts saved to: {plot_path}")

    return fig

def print_speed_summary(df):
    """Print summary of speed comparison results."""
    print("\n" + "=" * 60)
    print("🏁 SPEED COMPARISON SUMMARY")
    print("=" * 60)

    if df.empty:
        print("No valid results to summarize.")
        return

    quantum_df = df[df['model_type'] == 'quantum']
    classical_df = df[df['model_type'] == 'classical']

    if not quantum_df.empty and not classical_df.empty:
        # Average performance metrics
        avg_quantum_train = quantum_df['training_time'].mean()
        avg_classical_train = classical_df['training_time'].mean()

        avg_quantum_inference = quantum_df['inference_time'].mean() * 1000  # ms
        avg_classical_inference = classical_df['inference_time'].mean() * 1000  # ms

        print(f"📈 Average Training Time:")
        print(f"   Quantum: {avg_quantum_train:.2f}s")
        print(f"   Classical: {avg_classical_train:.2f}s")
        print(f"   Ratio (C/Q): {avg_classical_train/avg_quantum_train:.2f}x")

        print(f"\n⚡ Average Inference Time:")
        print(f"   Quantum: {avg_quantum_inference:.2f}ms")
        print(f"   Classical: {avg_classical_inference:.2f}ms")
        print(f"   Ratio (C/Q): {avg_classical_inference/avg_quantum_inference:.2f}x")

        print(f"\n🎯 Key Findings:")
        if avg_classical_train < avg_quantum_train:
            print(f"   • Classical training is {avg_quantum_train/avg_classical_train:.1f}x faster")
        else:
            print(f"   • Quantum training is {avg_classical_train/avg_quantum_train:.1f}x faster")

        if avg_classical_inference < avg_quantum_inference:
            print(f"   • Classical inference is {avg_quantum_inference/avg_classical_inference:.1f}x faster")
        else:
            print(f"   • Quantum inference is {avg_classical_inference/avg_quantum_inference:.1f}x faster")

    print(f"\n📊 Detailed Results:")
    print(df.to_string(index=False))

def main():
    """Run the quantum vs classical speed comparison."""
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("🚀 Starting Quantum vs Classical Speed Comparison")
    print("🔬 This will benchmark training and inference speeds...")

    # Run comparison
    df = run_speed_comparison(max_qubits=6, n_samples=80, epochs=3)

    if df.empty:
        print("❌ No successful benchmarks completed.")
        return

    # Save results
    csv_path = results_dir / "quantum_vs_classical_speed.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

    # Create charts
    create_speed_comparison_charts(df, results_dir)

    # Print summary
    print_speed_summary(df)

    return df

if __name__ == "__main__":
    df = main()