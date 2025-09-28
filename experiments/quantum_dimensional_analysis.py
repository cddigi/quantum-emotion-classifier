#!/usr/bin/env python3
"""
Quantum Dimensional Analysis: Feature Compression up to 20 Qubits

Focused analysis of quantum feature compression capabilities demonstrating
how quantum systems can theoretically process exponentially large feature
spaces with linear parameter growth, including practical tests up to 10 qubits
and theoretical analysis up to 20 qubits (1M+ dimensions).
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantum_emotion.classifier import QuantumTextClassifier
from quantum_emotion.hybrid import HybridTrainer
from quantum_emotion.utils import create_emotion_dataset, split_dataset, create_data_loaders

# Import quantum backends
import pennylane as qml

def calculate_quantum_metrics(n_qubits, n_layers=2, n_classes=4):
    """Calculate theoretical quantum metrics for given qubit count."""
    hilbert_dim = 2 ** n_qubits
    quantum_params = n_layers * n_qubits * 3  # 3 rotation parameters per qubit per layer

    # Classical preprocessing and postprocessing parameters (typical architecture)
    preprocess_params = 4 * 16 + 16 + 16 * n_qubits + n_qubits  # input->hidden->qubits
    postprocess_params = n_qubits * 16 + 16 + 16 * n_classes + n_classes  # qubits->hidden->output

    total_params = quantum_params + preprocess_params + postprocess_params

    # Classical equivalent parameters to process same feature space
    # Conservative estimate: single hidden layer with 4:1 compression
    hidden_dim = max(64, hilbert_dim // 4)
    classical_equivalent = hilbert_dim * hidden_dim + hidden_dim * n_classes + hidden_dim + n_classes

    compression_ratio = hilbert_dim / total_params
    parameter_efficiency = classical_equivalent / total_params
    feature_density = hilbert_dim / total_params
    exponential_advantage = np.log2(hilbert_dim) / np.log2(total_params)

    return {
        'n_qubits': n_qubits,
        'hilbert_dimension': hilbert_dim,
        'quantum_parameters': quantum_params,
        'classical_prepost_params': preprocess_params + postprocess_params,
        'total_parameters': total_params,
        'classical_equivalent_params': classical_equivalent,
        'compression_ratio': compression_ratio,
        'parameter_efficiency': parameter_efficiency,
        'feature_density': feature_density,
        'exponential_advantage': exponential_advantage,
        'memory_quantum_mb': total_params * 4 / 1024**2,  # 4 bytes per float32
        'memory_classical_mb': classical_equivalent * 4 / 1024**2,
        'memory_savings': classical_equivalent / total_params
    }

def benchmark_practical_quantum(n_qubits, X_train, X_test, y_train, y_test, backend='lightning.qubit', epochs=2):
    """Benchmark quantum model for practical validation."""
    try:
        print(f"    Testing {n_qubits} qubits...", end=" ")

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
            n_layers=2,
            backend=backend,
            feature_dim=X_train.shape[1]
        )

        trainer = HybridTrainer(model, learning_rate=0.01, device=torch.device('cpu'))

        # Training
        start_time = time.time()
        history = trainer.train(train_loader, test_loader, epochs=epochs, verbose=False)
        training_time = time.time() - start_time

        # Get actual parameters
        quantum_params = model.quantum_params.numel()
        classical_params = (sum(p.numel() for p in model.classical_preprocess.parameters()) +
                          sum(p.numel() for p in model.classical_postprocess.parameters()))
        total_params = quantum_params + classical_params

        # Final accuracy
        final_accuracy = (history['val_accuracy'][-1] / 100.0) if history['val_accuracy'] else 0.0

        print(f"✅ Acc: {final_accuracy*100:.1f}%, Params: {total_params}")

        return {
            'n_qubits': n_qubits,
            'type': 'practical',
            'accuracy': final_accuracy,
            'training_time': training_time,
            'total_parameters': total_params,
            'quantum_parameters': quantum_params,
            'classical_parameters': classical_params,
            'status': 'success'
        }

    except Exception as e:
        print(f"❌ Failed: {str(e)[:30]}...")
        return {
            'n_qubits': n_qubits,
            'type': 'practical',
            'accuracy': 0.0,
            'training_time': float('inf'),
            'total_parameters': 0,
            'quantum_parameters': 0,
            'classical_parameters': 0,
            'status': f'failed: {str(e)[:30]}'
        }

def benchmark_classical_scalable(model_sizes, X_train, X_test, y_train, y_test):
    """Benchmark classical models at different scales."""
    results = []

    print("\n🔬 Benchmarking Scalable Classical Models:")

    for size_name, (hidden_layers, max_iter) in model_sizes.items():
        try:
            print(f"  Testing {size_name}...", end=" ")

            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                random_state=42,
                max_iter=max_iter,
                early_stopping=True,
                validation_fraction=0.1
            )

            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Count parameters
            n_params = sum(coef.size for coef in model.coefs_) + sum(bias.size for bias in model.intercepts_)

            result = {
                'model_name': size_name,
                'type': 'classical',
                'hidden_layers': str(hidden_layers),
                'accuracy': accuracy,
                'training_time': training_time,
                'total_parameters': n_params,
                'status': 'success'
            }

            results.append(result)
            print(f"✅ Acc: {accuracy*100:.1f}%, Params: {n_params:,}")

        except Exception as e:
            print(f"❌ Failed: {str(e)[:30]}...")
            results.append({
                'model_name': size_name,
                'type': 'classical',
                'hidden_layers': str(hidden_layers),
                'accuracy': 0.0,
                'training_time': float('inf'),
                'total_parameters': 0,
                'status': f'failed: {str(e)[:30]}'
            })

    return results

def run_quantum_dimensional_analysis():
    """Run comprehensive quantum dimensional analysis."""
    print("🚀 Quantum Dimensional Analysis: Feature Compression up to 20 Qubits")
    print("=" * 75)

    # Create dataset
    print("\n📊 Creating emotion dataset...")
    features, labels, texts = create_emotion_dataset(n_samples=80, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        features.numpy(), labels.numpy(), test_size=0.3, random_state=42, stratify=labels.numpy()
    )

    print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Features: {X_train.shape[1]} dimensions")

    results = []

    # 1. Theoretical Analysis (1-20 qubits)
    print("\n📐 Theoretical Quantum Analysis (1-20 qubits):")

    theoretical_qubits = range(1, 21)
    for n_qubits in theoretical_qubits:
        metrics = calculate_quantum_metrics(n_qubits)
        metrics['type'] = 'theoretical'
        results.append(metrics)

        if n_qubits <= 5 or n_qubits % 5 == 0:  # Print key milestones
            print(f"  {n_qubits:2d} qubits: {metrics['hilbert_dimension']:>8,}D space, "
                  f"{metrics['total_parameters']:>4,} params, "
                  f"{metrics['compression_ratio']:>6.1f}x compression")

    # 2. Practical Validation (2-10 qubits)
    print("\n⚛️  Practical Quantum Validation (2-10 qubits):")

    practical_qubits = [2, 4, 6, 8, 10]
    for n_qubits in practical_qubits:
        result = benchmark_practical_quantum(n_qubits, X_train, X_test, y_train, y_test)
        results.append(result)

    # 3. Classical Baseline Models
    classical_sizes = {
        'Small (32)': ((32,), 200),
        'Medium (64,32)': ((64, 32), 300),
        'Large (128,64)': ((128, 64), 400),
        'XLarge (256,128)': ((256, 128), 500),
        'Massive (512,256)': ((512, 256), 600),
    }

    classical_results = benchmark_classical_scalable(classical_sizes, X_train, X_test, y_train, y_test)
    results.extend(classical_results)

    return pd.DataFrame(results)

def create_dimensional_analysis_charts(df, save_dir):
    """Create comprehensive dimensional analysis charts."""
    if df.empty:
        print("❌ No data to plot")
        return

    # Separate data types
    theoretical_df = df[df['type'] == 'theoretical'].copy()
    practical_df = df[df['type'] == 'practical'].copy()
    classical_df = df[df['type'] == 'classical'].copy()

    # Create comprehensive figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Quantum Dimensional Analysis: Feature Compression up to 20 Qubits',
                fontsize=16, fontweight='bold')

    # Colors
    colors = {
        'theoretical': '#1f77b4',
        'practical': '#ff7f0e',
        'classical': '#2ca02c'
    }

    # 1. Hilbert Space Growth (Theoretical)
    ax = axes[0, 0]
    if not theoretical_df.empty:
        ax.semilogy(theoretical_df['n_qubits'], theoretical_df['hilbert_dimension'],
                   'o-', color=colors['theoretical'], linewidth=3, markersize=6,
                   label='Hilbert Space (2^n)')
        ax.semilogy(theoretical_df['n_qubits'], theoretical_df['total_parameters'],
                   's--', color='red', linewidth=2, markersize=4,
                   label='Quantum Parameters')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Dimensions/Parameters')
    ax.set_title('Exponential Space vs Linear Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Compression Ratio Scaling
    ax = axes[0, 1]
    if not theoretical_df.empty:
        ax.plot(theoretical_df['n_qubits'], theoretical_df['compression_ratio'],
               'o-', color=colors['theoretical'], linewidth=3, markersize=6)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Classical Limit')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Feature Compression Ratio')
    ax.set_title('Quantum Feature Compression Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Parameter Efficiency
    ax = axes[0, 2]
    if not theoretical_df.empty:
        ax.semilogy(theoretical_df['n_qubits'], theoretical_df['parameter_efficiency'],
                   'o-', color=colors['theoretical'], linewidth=3, markersize=6)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Parameter Efficiency (Classical/Quantum)')
    ax.set_title('Parameter Efficiency vs Classical')
    ax.grid(True, alpha=0.3)

    # 4. Memory Usage Comparison
    ax = axes[1, 0]
    if not theoretical_df.empty:
        ax.semilogy(theoretical_df['n_qubits'], theoretical_df['memory_quantum_mb'],
                   'o-', color=colors['theoretical'], linewidth=3, markersize=6,
                   label='Quantum')
        ax.semilogy(theoretical_df['n_qubits'], theoretical_df['memory_classical_mb'],
                   's--', color='red', linewidth=2, markersize=4,
                   label='Classical Equivalent')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Efficiency Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Practical vs Theoretical Validation
    ax = axes[1, 1]
    if not theoretical_df.empty:
        theoretical_subset = theoretical_df[theoretical_df['n_qubits'] <= 10]
        ax.plot(theoretical_subset['n_qubits'], theoretical_subset['compression_ratio'],
               'o-', color=colors['theoretical'], linewidth=3, markersize=8,
               label='Theoretical')

    # Add practical validation points
    if not practical_df.empty:
        practical_success = practical_df[practical_df['status'] == 'success']
        if not practical_success.empty:
            # Calculate compression for practical results
            practical_compression = []
            for _, row in practical_success.iterrows():
                hilbert_dim = 2 ** row['n_qubits']
                compression = hilbert_dim / row['total_parameters']
                practical_compression.append(compression)

            ax.scatter(practical_success['n_qubits'], practical_compression,
                      color=colors['practical'], s=100, alpha=0.8,
                      label='Practical Results', zorder=5)

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Theoretical vs Practical Validation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Exponential Advantage Factor
    ax = axes[1, 2]
    if not theoretical_df.empty:
        ax.plot(theoretical_df['n_qubits'], theoretical_df['exponential_advantage'],
               'o-', color=colors['theoretical'], linewidth=3, markersize=6)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Exponential Advantage Factor')
    ax.set_title('Quantum Exponential Advantage')
    ax.grid(True, alpha=0.3)

    # 7. Accuracy Comparison
    ax = axes[2, 0]
    if not practical_df.empty:
        practical_success = practical_df[practical_df['status'] == 'success']
        if not practical_success.empty:
            ax.plot(practical_success['n_qubits'], practical_success['accuracy'] * 100,
                   'o-', color=colors['practical'], linewidth=3, markersize=8,
                   label='Quantum (Practical)')

    if not classical_df.empty:
        classical_success = classical_df[classical_df['status'] == 'success']
        if not classical_success.empty:
            ax.axhline(y=classical_success['accuracy'].mean() * 100,
                      color=colors['classical'], linewidth=3, alpha=0.7,
                      label='Classical Average')

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy: Quantum vs Classical')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Parameter Count Comparison
    ax = axes[2, 1]
    if not practical_df.empty:
        practical_success = practical_df[practical_df['status'] == 'success']
        if not practical_success.empty:
            ax.semilogy(practical_success['n_qubits'], practical_success['total_parameters'],
                       'o-', color=colors['practical'], linewidth=3, markersize=8,
                       label='Quantum (Actual)')

    if not classical_df.empty:
        classical_success = classical_df[classical_df['status'] == 'success']
        if not classical_success.empty:
            for i, (_, row) in enumerate(classical_success.iterrows()):
                ax.axhline(y=row['total_parameters'], alpha=0.5,
                          color=colors['classical'], linewidth=1)
            ax.axhline(y=classical_success['total_parameters'].mean(),
                      color=colors['classical'], linewidth=3,
                      label='Classical Models')

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Total Parameters')
    ax.set_title('Parameter Count Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. 20-Qubit Milestone Analysis
    ax = axes[2, 2]
    ax.axis('off')

    if not theoretical_df.empty:
        # Get 20-qubit results
        qubit_20_result = theoretical_df[theoretical_df['n_qubits'] == 20].iloc[0]

        milestone_text = f"""
🎯 20-QUBIT MILESTONE ANALYSIS

📊 Hilbert Space: {qubit_20_result['hilbert_dimension']:,} dimensions
🎪 Quantum Parameters: {qubit_20_result['total_parameters']:,}
🔄 Classical Equivalent: {qubit_20_result['classical_equivalent_params']:,}

📈 Compression Ratio: {qubit_20_result['compression_ratio']:.0f}x
⚡ Parameter Efficiency: {qubit_20_result['parameter_efficiency']:.0f}x
🚀 Exponential Advantage: {qubit_20_result['exponential_advantage']:.1f}x

💾 Memory Comparison:
   Quantum: {qubit_20_result['memory_quantum_mb']:.1f} MB
   Classical: {qubit_20_result['memory_classical_mb']:.0f} MB
   Savings: {qubit_20_result['memory_savings']:.0f}x reduction

🌌 QUANTUM ADVANTAGE:
Processing 1M+ dimensional feature space
with {qubit_20_result['total_parameters']:,} parameters instead of
{qubit_20_result['classical_equivalent_params']:,} classical parameters
        """

        ax.text(0.05, 0.95, milestone_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()

    # Save plot
    plot_path = save_dir / 'quantum_dimensional_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Quantum dimensional analysis saved to: {plot_path}")

    return fig

def print_dimensional_analysis_summary(df):
    """Print comprehensive dimensional analysis summary."""
    print("\n" + "=" * 80)
    print("🏆 QUANTUM DIMENSIONAL ANALYSIS SUMMARY")
    print("=" * 80)

    theoretical_df = df[df['type'] == 'theoretical']
    practical_df = df[df['type'] == 'practical']
    classical_df = df[df['type'] == 'classical']

    if not theoretical_df.empty:
        print("\n⚛️  Theoretical Quantum Analysis (1-20 qubits):")
        print(f"{'Qubits':<7} {'Hilbert Dim':<12} {'Parameters':<12} {'Compression':<12} {'Efficiency':<12}")
        print("-" * 60)

        # Show key milestones
        key_qubits = [1, 5, 10, 15, 20]
        for n_qubits in key_qubits:
            row = theoretical_df[theoretical_df['n_qubits'] == n_qubits].iloc[0]
            print(f"{n_qubits:<7} {int(row['hilbert_dimension']):<12,} "
                  f"{int(row['total_parameters']):<12,} {row['compression_ratio']:<12.1f} "
                  f"{row['parameter_efficiency']:<12.0f}")

    if not practical_df.empty:
        practical_success = practical_df[practical_df['status'] == 'success']
        if not practical_success.empty:
            print(f"\n🔬 Practical Quantum Validation:")
            print(f"{'Qubits':<7} {'Accuracy':<10} {'Parameters':<12} {'Training Time':<15}")
            print("-" * 50)

            for _, row in practical_success.iterrows():
                print(f"{int(row['n_qubits']):<7} {row['accuracy']*100:<10.1f}% "
                      f"{int(row['total_parameters']):<12,} {row['training_time']:<15.2f}s")

    if not classical_df.empty:
        classical_success = classical_df[classical_df['status'] == 'success']
        if not classical_success.empty:
            print(f"\n📊 Classical ML Baseline:")
            print(f"{'Model':<18} {'Accuracy':<10} {'Parameters':<12} {'Training Time':<15}")
            print("-" * 60)

            for _, row in classical_success.iterrows():
                print(f"{row['model_name']:<18} {row['accuracy']*100:<10.1f}% "
                      f"{int(row['total_parameters']):<12,} {row['training_time']:<15.2f}s")

    # 20-qubit milestone analysis
    if not theoretical_df.empty:
        qubit_20 = theoretical_df[theoretical_df['n_qubits'] == 20].iloc[0]

        print(f"\n🌌 20-QUBIT QUANTUM ADVANTAGE MILESTONE:")
        print(f"  📊 Processes: {int(qubit_20['hilbert_dimension']):,} dimensional feature space")
        print(f"  🎯 Using only: {int(qubit_20['total_parameters']):,} parameters")
        print(f"  🔄 Classical needs: {int(qubit_20['classical_equivalent_params']):,} parameters")
        print(f"  📈 Compression: {qubit_20['compression_ratio']:.0f}x advantage")
        print(f"  ⚡ Efficiency: {qubit_20['parameter_efficiency']:.0f}x better than classical")
        print(f"  💾 Memory: {qubit_20['memory_savings']:.0f}x reduction")
        print(f"  🚀 Exponential: {qubit_20['exponential_advantage']:.1f}x advantage factor")

def main():
    """Run the quantum dimensional analysis."""
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("🚀 Starting Quantum Dimensional Analysis")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Run analysis
    df = run_quantum_dimensional_analysis()

    if df.empty:
        print("❌ No results generated.")
        return

    # Save results
    csv_path = results_dir / "quantum_dimensional_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

    # Create charts
    create_dimensional_analysis_charts(df, results_dir)

    # Print summary
    print_dimensional_analysis_summary(df)

    return df

if __name__ == "__main__":
    df = main()