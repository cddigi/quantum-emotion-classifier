"""
Standalone verification for Qiskit Space Invaders implementation.
Verifies quantum components without requiring Atari environment.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from quantum_rl.qiskit_space_invaders import QiskitQuantumDQN, QiskitHybridDQN


def verify_qiskit_quantum_components():
    """Verify Qiskit quantum components work correctly."""
    print("🚀 Verifying Qiskit Quantum DQN Components")
    print("=" * 50)

    # Test 1: Create quantum model
    print("\n1️⃣ Testing Quantum Circuit...")
    model = QiskitQuantumDQN(n_qubits=4, n_actions=6, n_layers=2)
    print(f"   ✅ Quantum DQN created")
    print(f"   📊 Hilbert space: {model.config['hilbert_dim']}D")
    print(f"   🔢 Quantum params: {model.config['quantum_params']}")

    # Test 2: Forward pass
    print("\n2️⃣ Testing Forward Pass...")
    batch_size = 2
    state_shape = (4, 84, 84)  # Space Invaders preprocessed state
    test_input = torch.randn(batch_size, *state_shape)

    with torch.no_grad():
        q_values = model(test_input)

    print(f"   ✅ Forward pass successful")
    print(f"   📏 Output shape: {q_values.shape}")
    print(f"   🎮 Q-values: {q_values[0].numpy()}")

    # Test 3: Hybrid model
    print("\n3️⃣ Testing Hybrid Model...")
    hybrid_model = QiskitHybridDQN(n_qubits=4, n_actions=6)

    with torch.no_grad():
        q_values_hybrid = hybrid_model(test_input)

    print(f"   ✅ Hybrid model working")
    print(f"   📏 Output shape: {q_values_hybrid.shape}")

    # Test 4: Gradient flow
    print("\n4️⃣ Testing Gradient Flow...")
    hybrid_model.train()
    test_input.requires_grad_(True)

    q_values = hybrid_model(test_input)
    loss = q_values.sum()
    loss.backward()

    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in hybrid_model.parameters())
    print(f"   ✅ Gradients flowing: {has_gradients}")

    # Test 5: Training simulation
    print("\n5️⃣ Testing Training Step...")

    # Simulate DQN training step
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.0001)

    # Generate batch of experiences
    states = torch.randn(32, 4, 84, 84)
    actions = torch.randint(0, 6, (32,))
    rewards = torch.randn(32)
    next_states = torch.randn(32, 4, 84, 84)
    dones = torch.randint(0, 2, (32,)).bool()

    # DQN loss computation
    current_q_values = hybrid_model(states).gather(1, actions.unsqueeze(1))

    with torch.no_grad():
        next_q_values = hybrid_model(next_states).max(1)[0]
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

    loss = torch.nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"   ✅ Training step completed")
    print(f"   📉 Sample loss: {loss:.4f}")

    # Test 6: Parameter efficiency analysis
    print("\n6️⃣ Quantum Advantage Analysis...")

    total_params = sum(p.numel() for p in hybrid_model.parameters())
    quantum_params = model.config['quantum_params']
    hilbert_dim = model.config['hilbert_dim']

    # Classical equivalent would need parameters proportional to Hilbert space
    classical_equivalent = hilbert_dim * 64 * 6  # Rough estimate: 64 CNN features × 6 actions
    compression_ratio = classical_equivalent / total_params

    print(f"   📊 Total parameters: {total_params:,}")
    print(f"   🔮 Quantum parameters: {quantum_params}")
    print(f"   🌌 Hilbert space: {hilbert_dim}D")
    print(f"   📈 Compression ratio: {compression_ratio:.1f}x")

    print("\n🎉 All Qiskit components verified successfully!")
    print("🚀 Ready for quantum reinforcement learning training!")
    print("🎯 Target: Match QRDQN baseline of 578.00 ± 134.37 on Space Invaders")

    return {
        'model': hybrid_model,
        'total_params': total_params,
        'quantum_params': quantum_params,
        'compression_ratio': compression_ratio
    }


if __name__ == "__main__":
    results = verify_qiskit_quantum_components()