#!/usr/bin/env python3
"""
Qiskit Backend Performance Experiment

Tests quantum emotion classifier performance across different quantum backends
including PennyLane default and Qiskit simulators for 1-10 qubits.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantum_emotion.classifier import QuantumTextClassifier
from quantum_emotion.hybrid import HybridTrainer
from quantum_emotion.utils import create_emotion_dataset, split_dataset, create_data_loaders

# Import quantum backends
import pennylane as qml

def get_available_backends():
    """Get list of available quantum backends."""
    backends = {
        'default.qubit': 'PennyLane Default Simulator',
        'lightning.qubit': 'PennyLane Lightning (C++)',
    }

    # Try to add Qiskit backends
    try:
        import pennylane_qiskit
        backends.update({
            'qiskit.aer': 'Qiskit Aer Simulator',
            'qiskit.basicaer': 'Qiskit BasicAer Simulator',
        })
        print("✅ Qiskit backends available")
    except ImportError:
        print("❌ Qiskit backends not available")

    return backends

def test_backend_availability(backend_name, n_qubits=2):
    """Test if a backend is available and working."""
    try:
        # Create a simple test device
        dev = qml.device(backend_name, wires=n_qubits)

        @qml.qnode(dev)
        def test_circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        # Try to execute
        result = test_circuit()
        return True, None
    except Exception as e:
        return False, str(e)

def benchmark_backend(backend_name, n_qubits, n_samples=50, epochs=2):
    """Benchmark quantum classifier on a specific backend."""
    print(f"    Testing {backend_name} with {n_qubits} qubits...", end=" ")

    try:
        # Create dataset
        features, labels, _ = create_emotion_dataset(n_samples=n_samples)
        splits = split_dataset(features, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42)
        loaders = create_data_loaders(splits, batch_size=8)

        # Create model with specific backend
        model = QuantumTextClassifier(
            n_qubits=n_qubits,
            n_classes=4,
            n_layers=1,
            backend=backend_name,
            feature_dim=features.shape[1]
        )

        # Force CPU to ensure fair comparison
        trainer = HybridTrainer(model, learning_rate=0.01, device=torch.device('cpu'))

        # Measure circuit execution time (single forward pass)
        start_time = time.time()
        with torch.no_grad():
            sample_batch = next(iter(loaders['train']))[0][:4]  # Small batch
            _ = model(sample_batch)
        circuit_time = time.time() - start_time

        # Quick training benchmark
        start_time = time.time()
        history = trainer.train(
            loaders['train'],
            loaders['val'],
            epochs=epochs,
            verbose=False
        )
        training_time = time.time() - start_time

        # Get final accuracy
        final_accuracy = history['val_accuracy'][-1] if history['val_accuracy'] else 0.0

        # Calculate theoretical metrics
        hilbert_dim = 2 ** n_qubits
        quantum_params = model.quantum_params.numel()

        result = {
            'backend': backend_name,
            'n_qubits': n_qubits,
            'circuit_time': circuit_time,
            'training_time': training_time,
            'accuracy': final_accuracy,
            'hilbert_dimension': hilbert_dim,
            'quantum_parameters': quantum_params,
            'samples_per_second': (n_samples * epochs) / training_time,
            'status': 'success'
        }

        print(f"✅ {training_time:.2f}s, Acc: {final_accuracy*100:.1f}%")
        return result

    except Exception as e:
        print(f"❌ Failed: {str(e)[:50]}...")
        return {
            'backend': backend_name,
            'n_qubits': n_qubits,
            'circuit_time': float('inf'),
            'training_time': float('inf'),
            'accuracy': 0.0,
            'hilbert_dimension': 2 ** n_qubits,
            'quantum_parameters': 0,
            'samples_per_second': 0.0,
            'status': 'failed',
            'error': str(e)
        }

def run_backend_comparison(max_qubits=10, n_samples=50, epochs=2):
    """Run comprehensive backend comparison."""
    print("🔬 Quantum Backend Performance Comparison")
    print("=" * 60)

    # Get available backends
    available_backends = get_available_backends()
    print(f"📋 Available backends: {list(available_backends.keys())}")

    # Test backend availability
    working_backends = {}
    print("\n🧪 Testing backend availability...")
    for backend_name, description in available_backends.items():
        available, error = test_backend_availability(backend_name)
        if available:
            working_backends[backend_name] = description
            print(f"  ✅ {backend_name}: {description}")
        else:
            print(f"  ❌ {backend_name}: {error}")

    if not working_backends:
        print("❌ No working backends found!")
        return pd.DataFrame()

    print(f"\n🚀 Running performance tests on {len(working_backends)} backends...")

    results = []

    # Test each qubit count
    for n_qubits in range(1, min(max_qubits + 1, 11)):  # Limit to 10 qubits max
        print(f"\n📊 Testing {n_qubits} qubit(s):")

        for backend_name in working_backends.keys():
            # Skip problematic combinations
            if n_qubits > 8 and 'basicaer' in backend_name:
                print(f"    Skipping {backend_name} - too many qubits")
                continue

            result = benchmark_backend(backend_name, n_qubits, n_samples, epochs)
            results.append(result)

            # Add small delay between tests
            time.sleep(0.5)

    return pd.DataFrame(results)

def create_backend_comparison_charts(df, save_dir):
    """Create comprehensive backend comparison charts."""
    if df.empty:
        print("❌ No data to plot")
        return

    # Filter successful results
    success_df = df[df['status'] == 'success'].copy()

    if success_df.empty:
        print("❌ No successful results to plot")
        return

    # Get unique backends
    backends = success_df['backend'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(backends)))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quantum Backend Performance Comparison (1-10 Qubits)', fontsize=16, fontweight='bold')

    # 1. Training Time Comparison
    for i, backend in enumerate(backends):
        backend_data = success_df[success_df['backend'] == backend]
        axes[0, 0].plot(backend_data['n_qubits'], backend_data['training_time'],
                       'o-', linewidth=2, markersize=6, color=colors[i], label=backend)

    axes[0, 0].set_xlabel('Number of Qubits')
    axes[0, 0].set_ylabel('Training Time (seconds)')
    axes[0, 0].set_title('Training Time by Backend')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # 2. Circuit Execution Time
    for i, backend in enumerate(backends):
        backend_data = success_df[success_df['backend'] == backend]
        axes[0, 1].plot(backend_data['n_qubits'], backend_data['circuit_time'] * 1000,
                       'o-', linewidth=2, markersize=6, color=colors[i], label=backend)

    axes[0, 1].set_xlabel('Number of Qubits')
    axes[0, 1].set_ylabel('Circuit Time (milliseconds)')
    axes[0, 1].set_title('Circuit Execution Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Accuracy Comparison
    for i, backend in enumerate(backends):
        backend_data = success_df[success_df['backend'] == backend]
        axes[0, 2].plot(backend_data['n_qubits'], backend_data['accuracy'] * 100,
                       'o-', linewidth=2, markersize=6, color=colors[i], label=backend)

    axes[0, 2].set_xlabel('Number of Qubits')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Model Accuracy by Backend')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Throughput Comparison
    for i, backend in enumerate(backends):
        backend_data = success_df[success_df['backend'] == backend]
        axes[1, 0].plot(backend_data['n_qubits'], backend_data['samples_per_second'],
                       'o-', linewidth=2, markersize=6, color=colors[i], label=backend)

    axes[1, 0].set_xlabel('Number of Qubits')
    axes[1, 0].set_ylabel('Samples/Second')
    axes[1, 0].set_title('Training Throughput')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # 5. Backend Speed Heatmap
    if len(backends) > 1:
        pivot_df = success_df.pivot(index='n_qubits', columns='backend', values='training_time')
        im = axes[1, 1].imshow(pivot_df.values, cmap='RdYlGn_r', aspect='auto')
        axes[1, 1].set_xticks(range(len(backends)))
        axes[1, 1].set_xticklabels(backends, rotation=45)
        axes[1, 1].set_yticks(range(len(pivot_df.index)))
        axes[1, 1].set_yticklabels(pivot_df.index)
        axes[1, 1].set_title('Training Time Heatmap\n(Red=Slower, Green=Faster)')
        plt.colorbar(im, ax=axes[1, 1], label='Training Time (s)')

    # 6. Success Rate by Qubits
    success_rate = df.groupby('n_qubits')['status'].apply(lambda x: (x == 'success').mean() * 100)
    axes[1, 2].bar(success_rate.index, success_rate.values, color='skyblue', alpha=0.7)
    axes[1, 2].set_xlabel('Number of Qubits')
    axes[1, 2].set_ylabel('Success Rate (%)')
    axes[1, 2].set_title('Backend Success Rate')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = save_dir / 'qiskit_backend_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Backend comparison charts saved to: {plot_path}")

    return fig

def print_backend_summary(df):
    """Print summary of backend comparison results."""
    print("\n" + "=" * 70)
    print("🏆 BACKEND PERFORMANCE SUMMARY")
    print("=" * 70)

    if df.empty:
        print("No results to summarize.")
        return

    success_df = df[df['status'] == 'success']

    if success_df.empty:
        print("No successful runs to summarize.")
        return

    # Group by backend
    backend_summary = success_df.groupby('backend').agg({
        'training_time': ['mean', 'min', 'max'],
        'circuit_time': 'mean',
        'accuracy': 'mean',
        'n_qubits': ['min', 'max']
    }).round(4)

    print("📈 Performance by Backend:")
    print(backend_summary.to_string())

    # Find best performers
    best_speed = success_df.loc[success_df['training_time'].idxmin()]
    best_accuracy = success_df.loc[success_df['accuracy'].idxmax()]

    print(f"\n🏅 Best Performers:")
    print(f"  Fastest Training: {best_speed['backend']} ({best_speed['training_time']:.3f}s at {best_speed['n_qubits']} qubits)")
    print(f"  Highest Accuracy: {best_accuracy['backend']} ({best_accuracy['accuracy']*100:.1f}% at {best_accuracy['n_qubits']} qubits)")

    # Backend availability
    total_tests = len(df)
    successful_tests = len(success_df)
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Successful: {successful_tests}")
    print(f"  Success Rate: {successful_tests/total_tests*100:.1f}%")

def main():
    """Run the Qiskit backend performance experiment."""
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("🚀 Starting Qiskit Backend Performance Experiment")

    # Suppress some warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    # Run comparison
    df = run_backend_comparison(max_qubits=10, n_samples=40, epochs=2)

    if df.empty:
        print("❌ No results generated.")
        return

    # Save results
    csv_path = results_dir / "qiskit_backend_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

    # Create charts
    create_backend_comparison_charts(df, results_dir)

    # Print summary
    print_backend_summary(df)

    print(f"\n📋 Detailed Results:")
    print(df.to_string(index=False))

    return df

if __name__ == "__main__":
    df = main()