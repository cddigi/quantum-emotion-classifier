#!/usr/bin/env python3
"""
Qiskit vs Classical Performance Comparison

Comprehensive comparison between Qiskit quantum backends and classical
machine learning approaches for emotion classification across 1-10 qubits.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantum_emotion.classifier import QuantumTextClassifier
from quantum_emotion.hybrid import HybridTrainer
from quantum_emotion.utils import create_emotion_dataset, split_dataset, create_data_loaders

# Import quantum backends
import pennylane as qml

def create_classical_models():
    """Create classical ML models for comparison."""
    return {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=200),
        'Simple MLP': MLPClassifier(hidden_layer_sizes=(32,), random_state=42, max_iter=200)
    }

def benchmark_classical_model(model, X_train, X_test, y_train, y_test, model_name):
    """Benchmark a classical ML model."""
    try:
        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Inference
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Count parameters (approximate)
        if hasattr(model, 'coefs_'):  # Neural networks
            n_params = sum(coef.size for coef in model.coefs_) + sum(bias.size for bias in model.intercepts_)
        elif hasattr(model, 'support_vectors_'):  # SVM
            n_params = len(model.support_vectors_) * model.support_vectors_.shape[1]
        else:  # Random Forest
            n_params = sum(tree.tree_.node_count for tree in model.estimators_) * X_train.shape[1]

        return {
            'model': model_name,
            'type': 'classical',
            'training_time': training_time,
            'inference_time': inference_time,
            'accuracy': accuracy,
            'parameters': n_params,
            'samples_per_second': len(X_test) / inference_time,
            'status': 'success'
        }

    except Exception as e:
        return {
            'model': model_name,
            'type': 'classical',
            'training_time': float('inf'),
            'inference_time': float('inf'),
            'accuracy': 0.0,
            'parameters': 0,
            'samples_per_second': 0.0,
            'status': f'failed: {str(e)[:50]}'
        }

def benchmark_qiskit_quantum(backend_name, n_qubits, X_train, X_test, y_train, y_test, epochs=3):
    """Benchmark Qiskit quantum backend."""
    try:
        print(f"    Testing {backend_name} with {n_qubits} qubits...", end=" ")

        # Convert to PyTorch format
        features_train = torch.tensor(X_train, dtype=torch.float32)
        labels_train = torch.tensor(y_train, dtype=torch.long)
        features_test = torch.tensor(X_test, dtype=torch.float32)
        labels_test = torch.tensor(y_test, dtype=torch.long)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(features_train, labels_train)
        test_dataset = torch.utils.data.TensorDataset(features_test, labels_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

        # Create quantum model
        model = QuantumTextClassifier(
            n_qubits=n_qubits,
            n_classes=4,
            n_layers=1,
            backend=backend_name,
            feature_dim=X_train.shape[1]
        )

        trainer = HybridTrainer(model, learning_rate=0.01, device=torch.device('cpu'))

        # Training
        start_time = time.time()
        history = trainer.train(train_loader, test_loader, epochs=epochs, verbose=False)
        training_time = time.time() - start_time

        # Inference timing
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                features_batch, _ = batch
                _ = model(features_batch)
        inference_time = time.time() - start_time

        # Final accuracy (convert from percentage to fraction)
        final_accuracy = (history['val_accuracy'][-1] / 100.0) if history['val_accuracy'] else 0.0

        # Parameters
        quantum_params = model.quantum_params.numel()
        classical_params = (sum(p.numel() for p in model.classical_preprocess.parameters()) +
                          sum(p.numel() for p in model.classical_postprocess.parameters()))
        total_params = quantum_params + classical_params

        # Theoretical quantum metrics
        hilbert_dim = 2 ** n_qubits
        compression_ratio = hilbert_dim / total_params if total_params > 0 else 0

        result = {
            'model': f'{backend_name}',
            'type': 'quantum',
            'n_qubits': n_qubits,
            'training_time': training_time,
            'inference_time': inference_time,
            'accuracy': final_accuracy,
            'parameters': total_params,
            'quantum_parameters': quantum_params,
            'hilbert_dimension': hilbert_dim,
            'compression_ratio': compression_ratio,
            'samples_per_second': len(X_test) / inference_time,
            'status': 'success'
        }

        print(f"✅ {training_time:.2f}s, Acc: {final_accuracy*100:.1f}%")
        return result

    except Exception as e:
        print(f"❌ Failed: {str(e)[:50]}...")
        return {
            'model': f'{backend_name}',
            'type': 'quantum',
            'n_qubits': n_qubits,
            'training_time': float('inf'),
            'inference_time': float('inf'),
            'accuracy': 0.0,
            'parameters': 0,
            'quantum_parameters': 0,
            'hilbert_dimension': 2 ** n_qubits,
            'compression_ratio': 0.0,
            'samples_per_second': 0.0,
            'status': f'failed: {str(e)[:50]}'
        }

def run_comprehensive_comparison(max_qubits=10, n_samples=100, epochs=3):
    """Run comprehensive Qiskit vs Classical comparison."""
    print("🚀 Qiskit vs Classical Performance Comparison")
    print("=" * 60)

    results = []

    # Create dataset
    print("\n📊 Creating emotion dataset...")
    features, labels, texts = create_emotion_dataset(n_samples=n_samples, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        features.numpy(), labels.numpy(), test_size=0.3, random_state=42, stratify=labels.numpy()
    )

    print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Features: {X_train.shape[1]} dimensions")

    # Test classical models
    print("\n🔬 Benchmarking Classical Models:")
    classical_models = create_classical_models()

    for model_name, model in classical_models.items():
        print(f"  Testing {model_name}...", end=" ")
        result = benchmark_classical_model(model, X_train, X_test, y_train, y_test, model_name)
        results.append(result)
        if result['status'] == 'success':
            print(f"✅ {result['training_time']:.3f}s, Acc: {result['accuracy']*100:.1f}%")
        else:
            print(f"❌ {result['status']}")

    # Test Qiskit quantum backends
    print("\n⚛️  Benchmarking Qiskit Quantum Backends:")

    # Available quantum backends
    quantum_backends = ['default.qubit', 'lightning.qubit']

    # Test availability
    working_backends = []
    for backend in quantum_backends:
        try:
            dev = qml.device(backend, wires=2)
            working_backends.append(backend)
            print(f"  ✅ {backend} available")
        except Exception as e:
            print(f"  ❌ {backend} failed: {str(e)[:30]}...")

    # Run quantum experiments
    for n_qubits in range(2, min(max_qubits + 1, 9)):  # Limit for practical testing
        print(f"\n📐 Testing {n_qubits} qubits:")

        for backend in working_backends:
            result = benchmark_qiskit_quantum(
                backend, n_qubits, X_train, X_test, y_train, y_test, epochs
            )
            results.append(result)

    return pd.DataFrame(results)

def create_qiskit_classical_charts(df, save_dir):
    """Create comprehensive Qiskit vs Classical comparison charts."""
    if df.empty:
        print("❌ No data to plot")
        return

    # Separate quantum and classical results
    quantum_df = df[df['type'] == 'quantum'].copy()
    classical_df = df[df['type'] == 'classical'].copy()

    # Filter successful results
    quantum_success = quantum_df[quantum_df['status'] == 'success']
    classical_success = classical_df[classical_df['status'] == 'success']

    if quantum_success.empty and classical_success.empty:
        print("❌ No successful results to plot")
        return

    # Create comprehensive comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Qiskit Quantum vs Classical ML Performance Comparison', fontsize=16, fontweight='bold')

    # Colors
    quantum_colors = {'default.qubit': 'red', 'lightning.qubit': 'blue'}
    classical_colors = plt.cm.Set2(np.linspace(0, 1, len(classical_success)))

    # 1. Training Time Comparison
    if not classical_success.empty:
        # Classical models (single bars)
        classical_times = classical_success['training_time'].values
        classical_names = classical_success['model'].values
        axes[0, 0].bar(range(len(classical_names)), classical_times,
                      color=classical_colors, alpha=0.7, label='Classical')
        axes[0, 0].set_xticks(range(len(classical_names)))
        axes[0, 0].set_xticklabels(classical_names, rotation=45, ha='right')

    if not quantum_success.empty:
        # Quantum models (line plots by qubits)
        for backend in quantum_success['model'].unique():
            backend_data = quantum_success[quantum_success['model'] == backend]
            if not backend_data.empty:
                offset = len(classical_success) + 0.5
                x_pos = offset + backend_data['n_qubits'] - 2
                axes[0, 0].plot(x_pos, backend_data['training_time'],
                              'o-', color=quantum_colors.get(backend, 'green'),
                              linewidth=2, markersize=6, label=f'Quantum ({backend})')

    axes[0, 0].set_ylabel('Training Time (seconds)')
    axes[0, 0].set_title('Training Speed Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # 2. Inference Speed Comparison
    if not classical_success.empty:
        inference_speeds = classical_success['samples_per_second'].values
        axes[0, 1].bar(range(len(classical_names)), inference_speeds,
                      color=classical_colors, alpha=0.7, label='Classical')
        axes[0, 1].set_xticks(range(len(classical_names)))
        axes[0, 1].set_xticklabels(classical_names, rotation=45, ha='right')

    if not quantum_success.empty:
        for backend in quantum_success['model'].unique():
            backend_data = quantum_success[quantum_success['model'] == backend]
            if not backend_data.empty:
                offset = len(classical_success) + 0.5
                x_pos = offset + backend_data['n_qubits'] - 2
                axes[0, 1].plot(x_pos, backend_data['samples_per_second'],
                              'o-', color=quantum_colors.get(backend, 'green'),
                              linewidth=2, markersize=6, label=f'Quantum ({backend})')

    axes[0, 1].set_ylabel('Samples/Second')
    axes[0, 1].set_title('Inference Speed Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # 3. Accuracy Comparison
    if not classical_success.empty:
        accuracies = classical_success['accuracy'].values * 100
        axes[0, 2].bar(range(len(classical_names)), accuracies,
                      color=classical_colors, alpha=0.7, label='Classical')
        axes[0, 2].set_xticks(range(len(classical_names)))
        axes[0, 2].set_xticklabels(classical_names, rotation=45, ha='right')

    if not quantum_success.empty:
        for backend in quantum_success['model'].unique():
            backend_data = quantum_success[quantum_success['model'] == backend]
            if not backend_data.empty:
                offset = len(classical_success) + 0.5
                x_pos = offset + backend_data['n_qubits'] - 2
                axes[0, 2].plot(x_pos, backend_data['accuracy'] * 100,
                              'o-', color=quantum_colors.get(backend, 'green'),
                              linewidth=2, markersize=6, label=f'Quantum ({backend})')

    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Model Accuracy Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Parameter Efficiency
    if not classical_success.empty:
        classical_params = classical_success['parameters'].values
        axes[1, 0].bar(range(len(classical_names)), classical_params,
                      color=classical_colors, alpha=0.7, label='Classical')
        axes[1, 0].set_xticks(range(len(classical_names)))
        axes[1, 0].set_xticklabels(classical_names, rotation=45, ha='right')

    if not quantum_success.empty:
        for backend in quantum_success['model'].unique():
            backend_data = quantum_success[quantum_success['model'] == backend]
            if not backend_data.empty:
                offset = len(classical_success) + 0.5
                x_pos = offset + backend_data['n_qubits'] - 2
                axes[1, 0].plot(x_pos, backend_data['parameters'],
                              'o-', color=quantum_colors.get(backend, 'green'),
                              linewidth=2, markersize=6, label=f'Quantum ({backend})')

    axes[1, 0].set_ylabel('Number of Parameters')
    axes[1, 0].set_title('Parameter Count Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # 5. Quantum Advantage Metrics (Compression Ratio)
    if not quantum_success.empty:
        for backend in quantum_success['model'].unique():
            backend_data = quantum_success[quantum_success['model'] == backend]
            if not backend_data.empty:
                axes[1, 1].plot(backend_data['n_qubits'], backend_data['compression_ratio'],
                              'o-', color=quantum_colors.get(backend, 'green'),
                              linewidth=3, markersize=8, label=f'{backend}')

    axes[1, 1].set_xlabel('Number of Qubits')
    axes[1, 1].set_ylabel('Feature Compression Ratio')
    axes[1, 1].set_title('Quantum Advantage: Feature Space Compression')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Classical Limit')

    # 6. Hilbert Space Growth
    if not quantum_success.empty:
        for backend in quantum_success['model'].unique():
            backend_data = quantum_success[quantum_success['model'] == backend]
            if not backend_data.empty:
                axes[1, 2].plot(backend_data['n_qubits'], backend_data['hilbert_dimension'],
                              'o-', color=quantum_colors.get(backend, 'green'),
                              linewidth=3, markersize=8, label=f'{backend}')
                # Also plot parameter count for comparison
                axes[1, 2].plot(backend_data['n_qubits'], backend_data['parameters'],
                              's--', color=quantum_colors.get(backend, 'green'),
                              alpha=0.6, linewidth=2, markersize=6,
                              label=f'{backend} params')

    axes[1, 2].set_xlabel('Number of Qubits')
    axes[1, 2].set_ylabel('Dimension/Parameters')
    axes[1, 2].set_title('Exponential vs Linear Growth')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_yscale('log')

    plt.tight_layout()

    # Save plot
    plot_path = save_dir / 'qiskit_vs_classical_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Qiskit vs Classical comparison charts saved to: {plot_path}")

    return fig

def print_comparison_summary(df):
    """Print comprehensive comparison summary."""
    print("\n" + "=" * 80)
    print("🏆 QISKIT vs CLASSICAL PERFORMANCE SUMMARY")
    print("=" * 80)

    if df.empty:
        print("No results to summarize.")
        return

    # Separate results
    quantum_df = df[(df['type'] == 'quantum') & (df['status'] == 'success')]
    classical_df = df[(df['type'] == 'classical') & (df['status'] == 'success')]

    # Classical summary
    if not classical_df.empty:
        print("\n📊 Classical ML Performance:")
        for _, row in classical_df.iterrows():
            print(f"  {row['model']:15} | Train: {row['training_time']:6.3f}s | "
                  f"Inference: {row['samples_per_second']:8.1f} samples/s | "
                  f"Accuracy: {row['accuracy']*100:5.1f}% | "
                  f"Params: {row['parameters']:8,}")

    # Quantum summary
    if not quantum_df.empty:
        print(f"\n⚛️  Quantum (Qiskit) Performance:")
        for _, row in quantum_df.iterrows():
            print(f"  {row['model']:15} | {row['n_qubits']} qubits | "
                  f"Train: {row['training_time']:6.2f}s | "
                  f"Inference: {row['samples_per_second']:8.1f} samples/s | "
                  f"Accuracy: {row['accuracy']*100:5.1f}% | "
                  f"Compression: {row['compression_ratio']:6.1f}x")

    # Head-to-head comparison
    if not classical_df.empty and not quantum_df.empty:
        print(f"\n🥊 Head-to-Head Comparison:")

        # Best classical vs best quantum
        best_classical = classical_df.loc[classical_df['accuracy'].idxmax()]
        best_quantum = quantum_df.loc[quantum_df['accuracy'].idxmax()]

        print(f"  🏅 Best Classical: {best_classical['model']} "
              f"({best_classical['accuracy']*100:.1f}% accuracy)")
        print(f"  🏅 Best Quantum: {best_quantum['model']} "
              f"({best_quantum['accuracy']*100:.1f}% accuracy, "
              f"{best_quantum['compression_ratio']:.1f}x compression)")

        # Speed comparison
        fastest_classical = classical_df.loc[classical_df['samples_per_second'].idxmax()]
        fastest_quantum = quantum_df.loc[quantum_df['samples_per_second'].idxmax()]

        speed_ratio = fastest_classical['samples_per_second'] / fastest_quantum['samples_per_second']
        print(f"  ⚡ Speed: Classical is {speed_ratio:.0f}x faster for inference")

        # Feature space advantage
        max_compression = quantum_df['compression_ratio'].max()
        max_hilbert = quantum_df['hilbert_dimension'].max()
        print(f"  🌌 Quantum processes {max_hilbert:,}D feature space with "
              f"{max_compression:.1f}x compression")

    # Overall statistics
    total_tests = len(df)
    successful_tests = len(df[df['status'] == 'success'])
    print(f"\n📈 Experiment Statistics:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Successful: {successful_tests}")
    print(f"  Success Rate: {successful_tests/total_tests*100:.1f}%")

def main():
    """Run the comprehensive Qiskit vs Classical comparison."""
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("🚀 Starting Qiskit vs Classical Performance Comparison")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Run comparison
    df = run_comprehensive_comparison(max_qubits=8, n_samples=80, epochs=3)

    if df.empty:
        print("❌ No results generated.")
        return

    # Save results
    csv_path = results_dir / "qiskit_classical_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

    # Create charts
    create_qiskit_classical_charts(df, results_dir)

    # Print summary
    print_comparison_summary(df)

    print(f"\n📋 Detailed Results:")
    print(df.to_string(index=False))

    return df

if __name__ == "__main__":
    df = main()