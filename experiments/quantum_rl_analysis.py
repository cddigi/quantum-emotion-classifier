#!/usr/bin/env python3
"""
Quantum RL Analysis: Theoretical vs Practical Comparison

Analysis of quantum reinforcement learning capabilities compared to
classical approaches, including theoretical advantages and practical
implementation challenges for game environments like Space Invaders.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pennylane as qml
import torch
import torch.nn as nn


def analyze_quantum_rl_advantages():
    """Analyze theoretical quantum RL advantages."""

    print("🎮 Quantum Reinforcement Learning Analysis")
    print("=" * 50)

    # Theoretical analysis of quantum vs classical RL
    analysis_data = []

    # Different problem scales
    problem_scales = [
        {"name": "Simple (CartPole)", "state_dim": 4, "action_dim": 2, "complexity": "low"},
        {"name": "Atari (Space Invaders)", "state_dim": 84*84*4, "action_dim": 6, "complexity": "medium"},
        {"name": "Complex Game", "state_dim": 1000, "action_dim": 20, "complexity": "high"},
        {"name": "Massive RL Problem", "state_dim": 10000, "action_dim": 100, "complexity": "massive"}
    ]

    quantum_configs = [4, 6, 8, 10, 12, 16, 20]

    for problem in problem_scales:
        for n_qubits in quantum_configs:
            # Quantum analysis
            hilbert_dim = 2 ** n_qubits
            quantum_params = calculate_quantum_rl_params(n_qubits, problem["action_dim"])

            # Classical equivalent
            classical_params = calculate_classical_rl_params(problem["state_dim"], problem["action_dim"])

            # Quantum advantage metrics
            if quantum_params > 0 and hilbert_dim >= problem["state_dim"]:
                compression_ratio = hilbert_dim / quantum_params
                parameter_efficiency = classical_params / quantum_params
                memory_advantage = classical_params / quantum_params

                analysis_data.append({
                    'problem': problem["name"],
                    'complexity': problem["complexity"],
                    'state_dim': problem["state_dim"],
                    'action_dim': problem["action_dim"],
                    'n_qubits': n_qubits,
                    'hilbert_dimension': hilbert_dim,
                    'quantum_params': quantum_params,
                    'classical_params': classical_params,
                    'compression_ratio': compression_ratio,
                    'parameter_efficiency': parameter_efficiency,
                    'memory_advantage': memory_advantage,
                    'feasible': hilbert_dim >= problem["state_dim"]
                })

    return pd.DataFrame(analysis_data)


def calculate_quantum_rl_params(n_qubits, n_actions, n_layers=3):
    """Calculate quantum RL model parameters."""
    # Quantum circuit parameters
    quantum_circuit_params = n_layers * n_qubits * 3  # RX, RY, RZ per qubit per layer

    # Classical preprocessing and postprocessing
    preprocess_params = 64 * 32 + 32 + 32 * n_qubits + n_qubits  # Feature extraction -> quantum
    postprocess_params = n_qubits * 64 + 64 + 64 * n_actions + n_actions  # Quantum -> actions

    total_params = quantum_circuit_params + preprocess_params + postprocess_params
    return total_params


def calculate_classical_rl_params(state_dim, n_actions):
    """Calculate classical RL model parameters (CNN + DQN style)."""
    # CNN feature extraction (approximate for Atari-style)
    if state_dim > 1000:  # Large state space (like Atari)
        cnn_params = 32*8*8*4 + 64*4*4*32 + 64*3*3*64 + 512*64*7*7  # Typical CNN
        fc_params = 512 * 256 + 256 + 256 * n_actions + n_actions
    else:  # Small state space
        fc_params = state_dim * 128 + 128 + 128 * 64 + 64 + 64 * n_actions + n_actions
        cnn_params = 0

    return cnn_params + fc_params


def create_quantum_rl_advantage_charts(df, save_dir):
    """Create quantum RL advantage analysis charts."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quantum RL Advantages: Theoretical Analysis vs Classical Methods',
                fontsize=16, fontweight='bold')

    # Filter feasible quantum solutions
    feasible_df = df[df['feasible'] == True].copy()

    # Colors for different problem complexities
    complexity_colors = {
        'low': '#2ca02c',
        'medium': '#ff7f0e',
        'high': '#d62728',
        'massive': '#9467bd'
    }

    # 1. Parameter Efficiency by Problem Scale
    ax = axes[0, 0]
    for complexity in feasible_df['complexity'].unique():
        subset = feasible_df[feasible_df['complexity'] == complexity]
        if not subset.empty:
            ax.scatter(subset['n_qubits'], subset['parameter_efficiency'],
                      c=complexity_colors[complexity], label=complexity.title(),
                      s=100, alpha=0.7)

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Parameter Efficiency (Classical/Quantum)')
    ax.set_title('Parameter Efficiency vs Problem Scale')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Hilbert Space vs Parameters
    ax = axes[0, 1]
    for complexity in feasible_df['complexity'].unique():
        subset = feasible_df[feasible_df['complexity'] == complexity]
        if not subset.empty:
            ax.loglog(subset['quantum_params'], subset['hilbert_dimension'],
                     'o-', color=complexity_colors[complexity], label=complexity.title(),
                     markersize=6, alpha=0.7)

    # Add diagonal line for classical equivalent
    x_range = np.logspace(2, 6, 100)
    ax.loglog(x_range, x_range, 'k--', alpha=0.5, label='Classical Limit')

    ax.set_xlabel('Quantum Parameters')
    ax.set_ylabel('Hilbert Space Dimension')
    ax.set_title('Feature Space vs Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Memory Advantage by Qubit Count
    ax = axes[0, 2]
    space_invaders_df = feasible_df[feasible_df['problem'].str.contains('Space Invaders')]
    if not space_invaders_df.empty:
        ax.semilogy(space_invaders_df['n_qubits'], space_invaders_df['memory_advantage'],
                   'o-', color='red', linewidth=3, markersize=8, label='Space Invaders')

    ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Classical Baseline')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Memory Advantage (Classical/Quantum)')
    ax.set_title('Memory Efficiency: Space Invaders')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Compression Ratio Analysis
    ax = axes[1, 0]
    for complexity in feasible_df['complexity'].unique():
        subset = feasible_df[feasible_df['complexity'] == complexity]
        if not subset.empty:
            ax.plot(subset['n_qubits'], subset['compression_ratio'],
                   'o-', color=complexity_colors[complexity], label=complexity.title(),
                   linewidth=2, markersize=6)

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('Feature Compression by Problem Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Quantum Feasibility Matrix
    ax = axes[1, 1]

    # Create feasibility matrix
    problems = feasible_df['problem'].unique()
    qubits = sorted(feasible_df['n_qubits'].unique())

    feasibility_matrix = np.zeros((len(problems), len(qubits)))
    efficiency_matrix = np.zeros((len(problems), len(qubits)))

    for i, problem in enumerate(problems):
        for j, n_q in enumerate(qubits):
            subset = feasible_df[(feasible_df['problem'] == problem) & (feasible_df['n_qubits'] == n_q)]
            if not subset.empty:
                feasibility_matrix[i, j] = 1
                efficiency_matrix[i, j] = np.log10(subset.iloc[0]['parameter_efficiency'])

    # Use efficiency for color intensity
    im = ax.imshow(efficiency_matrix, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(qubits)))
    ax.set_xticklabels(qubits)
    ax.set_yticks(range(len(problems)))
    ax.set_yticklabels([p.replace(' (', '\n(') for p in problems])
    ax.set_xlabel('Number of Qubits')
    ax.set_title('Quantum Advantage Heatmap\n(Log Parameter Efficiency)')
    plt.colorbar(im, ax=ax, label='Log₁₀(Parameter Efficiency)')

    # 6. Space Invaders Specific Analysis
    ax = axes[1, 2]
    space_invaders_df = feasible_df[feasible_df['problem'].str.contains('Space Invaders')]

    if not space_invaders_df.empty:
        # Classical QRDQN baseline
        classical_performance = 578.0  # From user's model
        classical_params = space_invaders_df.iloc[0]['classical_params']

        # Theoretical quantum performance scaling
        quantum_performance_estimate = []
        for _, row in space_invaders_df.iterrows():
            # Theoretical performance based on compression ratio
            # Higher compression might enable better feature representation
            compression_bonus = min(2.0, 1 + np.log10(row['compression_ratio']) / 4)
            estimated_perf = classical_performance * compression_bonus
            quantum_performance_estimate.append(estimated_perf)

        ax.plot(space_invaders_df['n_qubits'], quantum_performance_estimate,
               'o-', color='blue', linewidth=3, markersize=8, label='Quantum Potential')

        ax.axhline(y=classical_performance, color='red', linestyle='--',
                  linewidth=2, label='Classical QRDQN (578)')

        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Estimated Performance')
        ax.set_title('Space Invaders: Quantum Potential')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = save_dir / 'quantum_rl_advantages_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Quantum RL analysis saved to: {plot_path}")

    return fig


def create_quantum_rl_architectures_comparison():
    """Compare different quantum RL architectures."""

    architectures = {
        "Pure Quantum DQN": {
            "description": "Full quantum circuit for value function approximation",
            "advantages": ["Exponential feature space", "Quantum superposition", "Interference effects"],
            "challenges": ["Circuit depth", "Noise sensitivity", "Limited classical preprocessing"],
            "best_for": "Small state spaces with quantum structure",
            "qubits_needed": "log₂(state_space)"
        },
        "Hybrid Quantum-Classical": {
            "description": "CNN features + quantum processing + classical output",
            "advantages": ["Best of both worlds", "Practical for large inputs", "Gradual quantum advantage"],
            "challenges": ["Classical bottleneck", "Limited quantum speedup", "Architecture complexity"],
            "best_for": "Large visual inputs like Atari games",
            "qubits_needed": "4-10 for feature processing"
        },
        "Quantum Policy Gradients": {
            "description": "Quantum circuits for policy parameterization",
            "advantages": ["Natural policy representation", "Quantum exploration", "Continuous actions"],
            "challenges": ["Gradient estimation", "Policy expressivity", "Training stability"],
            "best_for": "Continuous control problems",
            "qubits_needed": "Depends on action complexity"
        },
        "Quantum Actor-Critic": {
            "description": "Separate quantum circuits for actor and critic",
            "advantages": ["Stable training", "Quantum value estimation", "Policy optimization"],
            "challenges": ["Double quantum overhead", "Synchronization", "Complexity"],
            "best_for": "Complex sequential decision problems",
            "qubits_needed": "2 × problem complexity"
        }
    }

    return architectures


def print_quantum_rl_analysis_summary(df, architectures):
    """Print comprehensive quantum RL analysis summary."""

    print("\n" + "=" * 80)
    print("🎮 QUANTUM REINFORCEMENT LEARNING ANALYSIS SUMMARY")
    print("=" * 80)

    # Space Invaders specific analysis
    space_invaders_df = df[df['problem'].str.contains('Space Invaders') & df['feasible']]

    if not space_invaders_df.empty:
        print(f"\n🎯 Space Invaders Quantum RL Analysis:")
        print(f"   Classical QRDQN Baseline: 578.00 ± 134.37")
        print(f"   Classical Parameters: ~{space_invaders_df.iloc[0]['classical_params']:,}")

        # Show quantum options
        print(f"\n⚛️  Quantum RL Options for Space Invaders:")
        for _, row in space_invaders_df.iterrows():
            if row['n_qubits'] <= 12:  # Practical range
                print(f"   {row['n_qubits']:2d} qubits: {row['hilbert_dimension']:>8,}D space, "
                      f"{row['quantum_params']:>6,} params, "
                      f"{row['parameter_efficiency']:>6.0f}x efficiency")

        # Best configuration
        best_config = space_invaders_df.loc[space_invaders_df['parameter_efficiency'].idxmax()]
        print(f"\n🏆 Optimal Configuration for Space Invaders:")
        print(f"   Qubits: {best_config['n_qubits']}")
        print(f"   Hilbert Space: {best_config['hilbert_dimension']:,} dimensions")
        print(f"   Parameters: {best_config['quantum_params']:,} (vs {best_config['classical_params']:,} classical)")
        print(f"   Efficiency: {best_config['parameter_efficiency']:.0f}x better")
        print(f"   Memory Advantage: {best_config['memory_advantage']:.0f}x less memory")

    # Architecture comparison
    print(f"\n🏗️  Quantum RL Architecture Comparison:")
    for name, details in architectures.items():
        print(f"\n📊 {name}:")
        print(f"   Description: {details['description']}")
        print(f"   Best for: {details['best_for']}")
        print(f"   Qubits needed: {details['qubits_needed']}")
        print(f"   Key advantages: {', '.join(details['advantages'][:2])}")

    # Current challenges and future outlook
    print(f"\n🔬 Current Quantum RL Challenges:")
    print(f"   🎪 Hardware: Limited quantum computers, noise, decoherence")
    print(f"   💻 Software: Classical simulation overhead, gradient estimation")
    print(f"   🧠 Theory: Quantum advantage proofs, optimal architectures")
    print(f"   🎮 Practice: Environment integration, training stability")

    print(f"\n🚀 Future Quantum RL Potential:")
    print(f"   📈 Exponential feature spaces with linear parameters")
    print(f"   ⚡ Quantum speedups for specific problem structures")
    print(f"   🎯 Novel exploration strategies via quantum superposition")
    print(f"   🌌 Quantum interference for policy optimization")

    # Practical recommendations
    print(f"\n💡 Practical Recommendations:")
    print(f"   🥇 Start with hybrid approaches (4-8 qubits)")
    print(f"   🥈 Use quantum for feature processing, not full policy")
    print(f"   🥉 Focus on problems with natural quantum structure")
    print(f"   🎯 Target 10-20 qubit range for real quantum advantage")


def main():
    """Run quantum RL advantage analysis."""

    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("🚀 Starting Quantum RL Advantage Analysis")

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Run theoretical analysis
    df = analyze_quantum_rl_advantages()

    # Get architecture comparisons
    architectures = create_quantum_rl_architectures_comparison()

    # Save results
    csv_path = results_dir / "quantum_rl_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Analysis saved to: {csv_path}")

    # Create charts
    create_quantum_rl_advantage_charts(df, results_dir)

    # Print comprehensive summary
    print_quantum_rl_analysis_summary(df, architectures)

    return df, architectures


if __name__ == "__main__":
    df, architectures = main()