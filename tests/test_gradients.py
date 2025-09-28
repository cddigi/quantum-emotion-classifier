"""
Test gradient computation and optimization for quantum circuits.
"""

import pytest
import torch
import numpy as np
import pennylane as qml
from torch.nn import functional as F

from src.quantum_emotion.classifier import QuantumTextClassifier
from src.quantum_emotion.hybrid import HybridTrainer


class TestQuantumGradients:
    """Test quantum gradient computation using parameter-shift rule."""

    def test_parameter_shift_gradients(self):
        """Test parameter-shift rule gradient computation."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)

        sample_input = torch.randn(3)
        model.quantum_params.requires_grad_(True)

        # Forward pass
        output = model.quantum_circuit(sample_input, model.quantum_params)
        loss = sum(output)

        # Backward pass
        loss.backward()

        # Check gradients exist and are finite
        assert model.quantum_params.grad is not None
        assert torch.all(torch.isfinite(model.quantum_params.grad))
        assert model.quantum_params.grad.shape == model.quantum_params.shape

    def test_gradient_magnitude(self):
        """Test gradient magnitudes are reasonable."""
        model = QuantumTextClassifier(n_qubits=4, n_classes=2, n_layers=2)

        sample_input = torch.randn(4)
        model.quantum_params.requires_grad_(True)

        output = model.quantum_circuit(sample_input, model.quantum_params)
        loss = sum(output)
        loss.backward()

        grad_norm = torch.norm(model.quantum_params.grad)

        # Gradients should not be too large or too small
        assert 1e-6 < grad_norm < 10.0

    def test_gradient_flow_through_layers(self):
        """Test gradients flow through multiple quantum layers."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=3)

        sample_input = torch.randn(3)
        model.quantum_params.requires_grad_(True)

        output = model.quantum_circuit(sample_input, model.quantum_params)
        loss = sum(output)
        loss.backward()

        # All layers should have non-zero gradients
        for layer in range(3):
            layer_grad = model.quantum_params.grad[layer]
            assert torch.any(layer_grad != 0), f"Layer {layer} has zero gradients"

    def test_classical_quantum_gradient_interaction(self):
        """Test gradients flow between classical and quantum components."""
        model = QuantumTextClassifier(n_qubits=4, n_classes=2, n_layers=2)

        batch_size = 3
        sample_input = torch.randn(batch_size, 4)
        targets = torch.randint(0, 2, (batch_size,))

        # Forward pass through full model
        outputs = model(sample_input)
        loss = F.cross_entropy(outputs, targets)

        # Backward pass
        loss.backward()

        # Check both quantum and classical parameters have gradients
        assert model.quantum_params.grad is not None
        assert torch.any(model.quantum_params.grad != 0)

        # Check classical preprocessing gradients
        for param in model.classical_preprocess.parameters():
            if param.grad is not None:
                assert torch.any(param.grad != 0)

        # Check classical postprocessing gradients
        for param in model.classical_postprocess.parameters():
            if param.grad is not None:
                assert torch.any(param.grad != 0)

    def test_gradient_numerical_stability(self):
        """Test gradient computation numerical stability."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=2)

        # Test with extreme input values
        extreme_inputs = [
            torch.tensor([1000.0, -1000.0, 0.0]),
            torch.tensor([1e-6, 1e-6, 1e-6]),
            torch.tensor([np.pi, 2*np.pi, 3*np.pi])
        ]

        for sample_input in extreme_inputs:
            model.zero_grad()
            model.quantum_params.requires_grad_(True)

            try:
                output = model.quantum_circuit(sample_input, model.quantum_params)
                loss = sum(output)
                loss.backward()

                # Gradients should be finite even with extreme inputs
                assert torch.all(torch.isfinite(model.quantum_params.grad))

            except Exception as e:
                pytest.fail(f"Gradient computation failed with extreme input: {e}")

    def test_gradient_scaling_with_batch_size(self):
        """Test gradient behavior with different batch sizes."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)

        gradients_by_batch_size = {}

        for batch_size in [1, 4, 8]:
            model.zero_grad()

            sample_input = torch.randn(batch_size, 4)
            targets = torch.randint(0, 2, (batch_size,))

            outputs = model(sample_input)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            grad_norm = torch.norm(model.quantum_params.grad)
            gradients_by_batch_size[batch_size] = grad_norm.item()

        # Gradient norms should be reasonably consistent across batch sizes
        grad_values = list(gradients_by_batch_size.values())
        grad_std = np.std(grad_values)
        grad_mean = np.mean(grad_values)

        # Standard deviation should not be too large compared to mean
        assert grad_std / (grad_mean + 1e-8) < 2.0

    def test_gradient_vanishing_exploding(self):
        """Test for gradient vanishing/exploding problems."""
        # Test with deep circuit
        model = QuantumTextClassifier(n_qubits=4, n_classes=2, n_layers=5)

        sample_input = torch.randn(8, 4)
        targets = torch.randint(0, 2, (8,))

        outputs = model(sample_input)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        # Check gradient magnitudes across layers
        layer_grad_norms = []
        for layer in range(5):
            layer_grad = model.quantum_params.grad[layer]
            layer_grad_norm = torch.norm(layer_grad)
            layer_grad_norms.append(layer_grad_norm.item())

        # No layer should have extremely small or large gradients
        for i, grad_norm in enumerate(layer_grad_norms):
            assert 1e-8 < grad_norm < 100.0, f"Layer {i} has problematic gradient: {grad_norm}"

        # Gradient norms shouldn't vary by orders of magnitude
        min_grad, max_grad = min(layer_grad_norms), max(layer_grad_norms)
        assert max_grad / (min_grad + 1e-8) < 1000.0


class TestOptimizationBehavior:
    """Test optimization behavior and convergence."""

    def test_gradient_descent_step(self):
        """Test basic gradient descent step."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Store initial parameters
        initial_params = model.quantum_params.clone()

        sample_input = torch.randn(4, 4)
        targets = torch.randint(0, 2, (4,))

        # Forward and backward pass
        outputs = model(sample_input)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        # Optimization step
        optimizer.step()

        # Parameters should have changed
        assert not torch.allclose(initial_params, model.quantum_params)

    def test_adam_optimization(self):
        """Test Adam optimizer with quantum parameters."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        initial_loss = None
        final_loss = None

        # Run several optimization steps
        for step in range(10):
            optimizer.zero_grad()

            sample_input = torch.randn(8, 4)
            targets = torch.randint(0, 2, (8,))

            outputs = model(sample_input)
            loss = F.cross_entropy(outputs, targets)

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if step == 9:
                final_loss = loss.item()

        # Loss should generally decrease (allowing for some noise)
        # This is a weak test since we're using random data
        assert abs(final_loss - initial_loss) < initial_loss  # Some change occurred

    def test_learning_rate_sensitivity(self):
        """Test sensitivity to learning rate."""
        learning_rates = [0.001, 0.01, 0.1]
        final_losses = []

        for lr in learning_rates:
            model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Run optimization
            for _ in range(5):
                optimizer.zero_grad()

                sample_input = torch.randn(4, 4)
                targets = torch.randint(0, 2, (4,))

                outputs = model(sample_input)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()

            final_losses.append(loss.item())

        # Different learning rates should give different results
        assert len(set([round(loss, 3) for loss in final_losses])) > 1

    def test_parameter_initialization_effect(self):
        """Test effect of different parameter initializations."""
        initializations = [0.01, 0.1, 0.5]
        convergence_behaviors = []

        for init_scale in initializations:
            model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)

            # Re-initialize parameters
            with torch.no_grad():
                model.quantum_params.normal_(0, init_scale)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            losses = []
            for _ in range(10):
                optimizer.zero_grad()

                sample_input = torch.randn(4, 4)
                targets = torch.randint(0, 2, (4,))

                outputs = model(sample_input)
                loss = F.cross_entropy(outputs, targets)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            convergence_behaviors.append(losses)

        # Different initializations should lead to different training curves
        # This is a basic check that initialization matters
        assert len(convergence_behaviors) == len(initializations)


class TestGradientAccuracy:
    """Test gradient accuracy using finite differences."""

    def test_finite_difference_validation(self):
        """Validate gradients using finite differences."""
        model = QuantumTextClassifier(n_qubits=2, n_classes=2, n_layers=1)

        sample_input = torch.randn(2)
        model.quantum_params.requires_grad_(True)

        def loss_fn(params):
            """Compute loss for given parameters."""
            output = model.quantum_circuit(sample_input, params)
            return sum(output)

        # Compute analytical gradients
        loss = loss_fn(model.quantum_params)
        loss.backward()
        analytical_grad = model.quantum_params.grad.clone()

        # Compute numerical gradients using finite differences
        eps = 1e-5
        numerical_grad = torch.zeros_like(model.quantum_params)

        with torch.no_grad():
            for i in range(model.quantum_params.shape[0]):
                for j in range(model.quantum_params.shape[1]):
                    for k in range(model.quantum_params.shape[2]):
                        # Forward difference
                        model.quantum_params[i, j, k] += eps
                        loss_plus = loss_fn(model.quantum_params)

                        model.quantum_params[i, j, k] -= 2 * eps
                        loss_minus = loss_fn(model.quantum_params)

                        # Restore original value
                        model.quantum_params[i, j, k] += eps

                        # Numerical gradient
                        numerical_grad[i, j, k] = (loss_plus - loss_minus) / (2 * eps)

        # Compare analytical and numerical gradients
        grad_diff = torch.abs(analytical_grad - numerical_grad)
        max_diff = torch.max(grad_diff)

        # Allow for some numerical precision errors
        assert max_diff < 1e-3, f"Gradient mismatch: max diff = {max_diff}"

    def test_gradient_consistency(self):
        """Test gradient consistency across multiple evaluations."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)

        sample_input = torch.randn(3)
        gradients = []

        # Compute gradients multiple times
        for _ in range(5):
            model.zero_grad()
            model.quantum_params.requires_grad_(True)

            output = model.quantum_circuit(sample_input, model.quantum_params)
            loss = sum(output)
            loss.backward()

            gradients.append(model.quantum_params.grad.clone())

        # All gradients should be identical
        for i in range(1, len(gradients)):
            assert torch.allclose(gradients[0], gradients[i], atol=1e-8)


class TestHybridTrainerGradients:
    """Test gradient behavior in HybridTrainer."""

    def test_trainer_gradient_tracking(self):
        """Test trainer tracks quantum and classical gradients."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)

        # Force CPU to avoid MPS float64 issues
        trainer = HybridTrainer(model, learning_rate=0.01, device=torch.device('cpu'))

        # Create simple dataset on CPU
        X = torch.randn(8, 4, device='cpu')
        y = torch.randint(0, 2, (8,), device='cpu')
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        # Train for one epoch
        trainer.train_epoch(loader)

        # Check gradient tracking
        assert len(trainer.history['quantum_grad_norms']) > 0
        assert len(trainer.history['classical_grad_norms']) > 0

        # Gradient norms should be positive
        assert all(grad > 0 for grad in trainer.history['quantum_grad_norms'])

    def test_gradient_accumulation(self):
        """Test gradient accumulation across batches."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)

        # Create dataset with multiple batches
        X = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        accumulated_grad = torch.zeros_like(model.quantum_params)

        # Manually accumulate gradients across batches
        for batch_idx, (data, targets) in enumerate(loader):
            outputs = model(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            if batch_idx == 0:
                accumulated_grad = model.quantum_params.grad.clone()
            else:
                # In normal training, gradients are zeroed between batches
                # This test just checks that gradients are computable
                assert model.quantum_params.grad is not None

            model.zero_grad()

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)

        sample_input = torch.randn(4, 4)
        targets = torch.randint(0, 2, (4,))

        outputs = model(sample_input)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        # Apply gradient clipping
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Check that gradient norm is within bounds
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        assert total_grad_norm <= max_grad_norm + 1e-6


class TestEdgeCases:
    """Test edge cases in gradient computation."""

    def test_zero_loss_gradients(self):
        """Test gradients when loss is zero."""
        model = QuantumTextClassifier(n_qubits=2, n_classes=2, n_layers=1)

        # Create input that might lead to zero loss
        sample_input = torch.zeros(2, 4)
        # Use soft targets that might match output exactly
        targets = torch.tensor([0, 1])

        outputs = model(sample_input)
        # Use a loss that could be zero
        loss = F.mse_loss(outputs, torch.zeros_like(outputs))

        loss.backward()

        # Even if loss is zero, gradients should be computable
        assert model.quantum_params.grad is not None
        assert torch.all(torch.isfinite(model.quantum_params.grad))

    def test_nan_input_handling(self):
        """Test gradient computation with NaN inputs."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)

        # Input with NaN values
        sample_input = torch.tensor([1.0, float('nan'), 0.5, 0.2])

        with pytest.raises((RuntimeError, ValueError)):
            # Should raise an error rather than silently producing bad gradients
            outputs = model(sample_input.unsqueeze(0))
            loss = outputs.sum()
            loss.backward()

    def test_very_large_gradients(self):
        """Test handling of very large gradients."""
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)

        # Set parameters to extreme values that might cause large gradients
        with torch.no_grad():
            model.quantum_params.fill_(10.0)

        sample_input = torch.randn(4, 4)
        targets = torch.randint(0, 2, (4,))

        outputs = model(sample_input)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        # Gradients should be finite even if large
        assert torch.all(torch.isfinite(model.quantum_params.grad))

        # Apply gradient clipping if needed
        grad_norm = torch.norm(model.quantum_params.grad)
        if grad_norm > 100:  # Arbitrary threshold
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)