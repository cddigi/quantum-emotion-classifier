"""
Test suite for Qiskit quantum emotion classifier.

Tests the Qiskit implementation of quantum circuits and emotion classification.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from quantum_emotion.qiskit_core import (
    QiskitEmotionClassifier,
    QiskitQuantumTrainer,
    create_quantum_feature_map,
    create_variational_ansatz,
    create_efficient_quantum_circuit
)
from quantum_emotion.utils import create_emotion_dataset, create_data_loaders, split_dataset


class TestQiskitEmotionClassifier:
    """Test Qiskit emotion classifier functionality."""

    def test_initialization(self):
        """Test model initialization with different configurations."""
        # Test default initialization
        model = QiskitEmotionClassifier()
        assert model.n_qubits == 6
        assert model.n_classes == 4
        assert model.n_layers == 3
        assert model.config['hilbert_dim'] == 64

        # Test custom initialization
        model = QiskitEmotionClassifier(n_qubits=4, n_classes=3, n_layers=2)
        assert model.n_qubits == 4
        assert model.n_classes == 3
        assert model.n_layers == 2
        assert model.config['hilbert_dim'] == 16

    def test_quantum_circuit_structure(self):
        """Test quantum circuit structure and properties."""
        model = QiskitEmotionClassifier(n_qubits=4, n_layers=2)

        # Test circuit depth
        depth = model.get_circuit_depth()
        assert depth > 0
        assert depth <= 20  # Reasonable depth for 4 qubits, 2 layers

        # Test parameter count
        n_params = model.config['n_quantum_params']
        expected_params = 2 * 4 * 3  # n_layers * n_qubits * 3 (rx, ry, rz)
        assert n_params == expected_params

    def test_forward_pass(self):
        """Test forward pass through the quantum-classical network."""
        model = QiskitEmotionClassifier(n_qubits=2, n_classes=4, n_layers=1)

        # Create test input
        batch_size = 2
        feature_dim = 4
        x = torch.randn(batch_size, feature_dim)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Check output shape
        assert output.shape == (batch_size, 4)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test gradient computation through quantum circuit."""
        model = QiskitEmotionClassifier(n_qubits=2, n_classes=4, n_layers=1)

        # Create test input
        x = torch.randn(2, 4, requires_grad=True)
        target = torch.tensor([0, 1])

        # Forward pass
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)

        # Check if gradients can be computed
        loss.backward()
        assert model.quantum_weights.grad is not None

    def test_circuit_execution(self):
        """Test quantum circuit execution with different inputs."""
        model = QiskitEmotionClassifier(n_qubits=3, n_layers=1)

        # Test data
        input_data = np.array([0.5, -0.5, 0.0])
        weights = np.random.randn(len(model.weight_params))

        # Execute circuit
        expectations = model.execute_quantum_circuit(input_data, weights)

        # Check output
        assert len(expectations) == 3
        assert all(-1 <= exp <= 1 for exp in expectations)

    def test_different_backends(self):
        """Test model with different Qiskit backends."""
        backends = ["aer_simulator"]

        for backend_name in backends:
            model = QiskitEmotionClassifier(
                n_qubits=2,
                n_classes=4,
                backend_name=backend_name
            )
            assert model.config['backend'] == backend_name

            # Test forward pass
            x = torch.randn(1, 4)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 4)


class TestQuantumCircuitComponents:
    """Test individual quantum circuit components."""

    def test_quantum_feature_map(self):
        """Test quantum feature map creation."""
        for n_qubits in [2, 4, 6]:
            feature_map = create_quantum_feature_map(n_qubits=n_qubits, reps=2)
            assert feature_map.num_qubits == n_qubits
            assert feature_map.num_parameters > 0

    def test_variational_ansatz(self):
        """Test variational ansatz creation."""
        for n_qubits in [2, 4, 6]:
            for n_layers in [1, 2, 3]:
                ansatz = create_variational_ansatz(n_qubits=n_qubits, n_layers=n_layers)
                assert ansatz.num_qubits == n_qubits
                assert ansatz.num_parameters > 0

    def test_efficient_quantum_circuit(self):
        """Test EfficientSU2 circuit creation."""
        for n_qubits in [2, 4, 6]:
            circuit = create_efficient_quantum_circuit(n_qubits=n_qubits, n_layers=2)
            assert circuit.num_qubits == n_qubits
            assert circuit.num_parameters > 0


class TestQiskitQuantumTrainer:
    """Test Qiskit quantum trainer functionality."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = QiskitEmotionClassifier(n_qubits=2, n_classes=4)
        trainer = QiskitQuantumTrainer(model, learning_rate=0.01)

        assert trainer.model == model
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_training_step(self):
        """Test single training epoch."""
        # Create simple model
        model = QiskitEmotionClassifier(n_qubits=2, n_classes=4, n_layers=1)
        trainer = QiskitQuantumTrainer(model, learning_rate=0.1)

        # Create small dataset
        features, labels, _ = create_emotion_dataset(n_samples=8)
        splits = split_dataset(features, labels, train_ratio=0.75, val_ratio=0.25)
        loaders = create_data_loaders(splits, batch_size=4)

        # Train for one epoch
        train_loss, train_acc = trainer.train_epoch(loaders['train'])

        assert train_loss > 0
        assert 0 <= train_acc <= 100

    def test_validation(self):
        """Test validation functionality."""
        model = QiskitEmotionClassifier(n_qubits=2, n_classes=4, n_layers=1)
        trainer = QiskitQuantumTrainer(model)

        # Create small dataset
        features, labels, _ = create_emotion_dataset(n_samples=8)
        splits = split_dataset(features, labels, train_ratio=0.75, val_ratio=0.25)
        loaders = create_data_loaders(splits, batch_size=4)

        # Validate
        val_loss, val_acc = trainer.validate(loaders['val'])

        assert val_loss > 0
        assert 0 <= val_acc <= 100

    def test_full_training(self):
        """Test full training pipeline."""
        model = QiskitEmotionClassifier(n_qubits=2, n_classes=4, n_layers=1)
        trainer = QiskitQuantumTrainer(model, learning_rate=0.1)

        # Create small dataset
        features, labels, _ = create_emotion_dataset(n_samples=16)
        splits = split_dataset(features, labels, train_ratio=0.75, val_ratio=0.25)
        loaders = create_data_loaders(splits, batch_size=4)

        # Train for a few epochs
        history = trainer.train(
            loaders['train'],
            loaders['val'],
            epochs=2,
            verbose=False
        )

        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2
        assert all(loss > 0 for loss in history['train_loss'])
        assert all(0 <= acc <= 100 for acc in history['train_accuracy'])


class TestQuantumAdvantages:
    """Test quantum advantage characteristics."""

    def test_exponential_feature_space(self):
        """Test exponential scaling of feature space."""
        qubit_counts = [2, 4, 6, 8]

        for n_qubits in qubit_counts:
            model = QiskitEmotionClassifier(n_qubits=n_qubits)
            hilbert_dim = model.config['hilbert_dim']
            n_params = model.config['n_quantum_params']

            # Check exponential feature space
            assert hilbert_dim == 2 ** n_qubits

            # Check linear parameter growth
            assert n_params == model.n_layers * n_qubits * 3

            # Check compression ratio
            if n_qubits >= 4:
                compression_ratio = hilbert_dim / n_params
                assert compression_ratio > 1

    def test_parameter_efficiency(self):
        """Test parameter efficiency compared to classical equivalent."""
        model = QiskitEmotionClassifier(n_qubits=6, n_classes=4)

        # Quantum parameters
        quantum_params = sum(p.numel() for p in model.parameters())

        # Classical equivalent would need parameters proportional to Hilbert space
        hilbert_dim = model.config['hilbert_dim']
        classical_equivalent = hilbert_dim * model.n_classes  # Simplified estimate

        # Check quantum advantage
        efficiency = classical_equivalent / quantum_params
        assert efficiency > 1  # Quantum should be more efficient


class TestCircuitVisualization:
    """Test circuit visualization and analysis."""

    def test_circuit_diagram(self):
        """Test circuit diagram generation."""
        model = QiskitEmotionClassifier(n_qubits=3, n_layers=1)
        diagram = model.get_circuit_diagram()

        assert isinstance(diagram, str)
        assert len(diagram) > 0
        assert 'q' in diagram  # Should show qubits
        assert 'H' in diagram  # Should show Hadamard gates

    def test_circuit_depth_scaling(self):
        """Test circuit depth scales with layers."""
        depths = []

        for n_layers in [1, 2, 3]:
            model = QiskitEmotionClassifier(n_qubits=3, n_layers=n_layers)
            depths.append(model.get_circuit_depth())

        # Depth should increase with layers
        assert depths[0] < depths[1] < depths[2]


if __name__ == "__main__":
    """Run basic tests to verify Qiskit implementation."""
    print("🧪 Testing Qiskit Quantum Emotion Classifier")
    print("=" * 50)

    # Test initialization
    print("\n✅ Testing initialization...")
    model = QiskitEmotionClassifier(n_qubits=4, n_classes=4, n_layers=2)
    print(f"   Model created with {model.n_qubits} qubits")

    # Test circuit
    print("\n✅ Testing circuit properties...")
    depth = model.get_circuit_depth()
    print(f"   Circuit depth: {depth}")

    # Test forward pass
    print("\n✅ Testing forward pass...")
    x = torch.randn(2, 4)
    with torch.no_grad():
        output = model(x)
    print(f"   Output shape: {output.shape}")

    # Test training
    print("\n✅ Testing training...")
    trainer = QiskitQuantumTrainer(model)
    features, labels, _ = create_emotion_dataset(n_samples=40)
    splits = split_dataset(features, labels, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    loaders = create_data_loaders(splits, batch_size=4)

    history = trainer.train(loaders['train'], loaders['val'], epochs=1, verbose=False)
    print(f"   Training loss: {history['train_loss'][0]:.4f}")

    print("\n🎉 All basic tests passed! Qiskit implementation working correctly.")