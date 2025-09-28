#!/usr/bin/env python3
"""
Quick Quantum Emotion Classifier Performance Test
Tests 1-10 qubits with minimal training for rapid results.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantum_emotion.classifier import QuantumTextClassifier
from quantum_emotion.utils import create_emotion_dataset

def quick_benchmark(n_qubits):
    """Quick benchmark for a specific qubit count."""
    print(f"Testing {n_qubits} qubit(s)...", end=" ")

    try:
        # Create minimal dataset
        features = torch.randn(20, 4)  # 20 samples, 4 features

        # Create model - CPU for consistency
        model = QuantumTextClassifier(
            n_qubits=n_qubits,
            n_classes=2,
            n_layers=1,
            feature_dim=4
        )

        # Time a single forward pass
        start = time.time()
        with torch.no_grad():
            for _ in range(5):  # Average over 5 runs
                _ = model(features)
        forward_time = (time.time() - start) / 5

        # Calculate metrics
        hilbert_dim = 2 ** n_qubits
        quantum_params = model.quantum_params.numel()
        compression = hilbert_dim / quantum_params if quantum_params > 0 else 0

        print(f"✓ Time: {forward_time:.3f}s, Compression: {compression:.1f}x")

        return {
            'n_qubits': n_qubits,
            'forward_time': forward_time,
            'hilbert_dimension': hilbert_dim,
            'quantum_parameters': quantum_params,
            'compression_ratio': compression
        }

    except Exception as e:
        print(f"✗ Failed: {e}")
        return None

def main():
    print("🚀 Quick Quantum Scaling Test (1-10 qubits)")
    print("=" * 50)

    results = []

    # Test 1-10 qubits
    for n in range(1, 11):
        result = quick_benchmark(n)
        if result:
            results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    df.to_csv(results_dir / "quick_scaling_results.csv", index=False)

    # Create simple plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Forward time
    axes[0].plot(df['n_qubits'], df['forward_time'], 'o-', color='blue')
    axes[0].set_xlabel('Number of Qubits')
    axes[0].set_ylabel('Forward Pass Time (s)')
    axes[0].set_title('Computation Time Scaling')
    axes[0].grid(True, alpha=0.3)

    # 2. Hilbert dimension vs parameters
    axes[1].semilogy(df['n_qubits'], df['hilbert_dimension'], 'o-', label='Hilbert Dim (2^n)', color='green')
    axes[1].plot(df['n_qubits'], df['quantum_parameters'], 's-', label='Parameters (linear)', color='orange')
    axes[1].set_xlabel('Number of Qubits')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Exponential vs Linear Growth')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Compression ratio
    axes[2].semilogy(df['n_qubits'], df['compression_ratio'], 'o-', color='purple')
    axes[2].set_xlabel('Number of Qubits')
    axes[2].set_ylabel('Compression Ratio')
    axes[2].set_title('Quantum Advantage Factor')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Quantum Emotion Classifier: Qubit Scaling Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_path = results_dir / "quick_scaling_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {plot_path}")

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print(f"✓ Tested {len(df)} qubit configurations")
    print(f"✓ Max compression ratio: {df['compression_ratio'].max():.1f}x at {df.loc[df['compression_ratio'].idxmax(), 'n_qubits']:.0f} qubits")
    print(f"✓ Hilbert space at 10 qubits: {df.loc[df['n_qubits']==10, 'hilbert_dimension'].values[0] if 10 in df['n_qubits'].values else 'N/A'}")

    print("\n📊 Results Table:")
    print(df.to_string(index=False))

    plt.show()

    return df

if __name__ == "__main__":
    df = main()