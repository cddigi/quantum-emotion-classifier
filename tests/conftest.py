"""
Pytest configuration and shared fixtures for quantum emotion classifier tests.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from typing import Tuple, List

from src.quantum_emotion.classifier import QuantumTextClassifier
from src.quantum_emotion.utils import create_emotion_dataset


@pytest.fixture
def sample_features():
    """Create sample feature tensor for testing."""
    return torch.randn(10, 4)


@pytest.fixture
def sample_labels():
    """Create sample label tensor for testing."""
    return torch.randint(0, 4, (10,))


@pytest.fixture
def emotion_names():
    """Provide standard emotion class names."""
    return ["Happy", "Sad", "Angry", "Neutral"]


@pytest.fixture
def small_model():
    """Create small quantum classifier for testing."""
    return QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)


@pytest.fixture
def sample_dataset():
    """Create sample emotion dataset."""
    features, labels, emotion_names = create_emotion_dataset(n_samples=20, random_state=42)
    return features, labels, emotion_names


@pytest.fixture
def binary_dataset():
    """Create binary classification dataset."""
    features, labels, emotion_names = create_emotion_dataset(n_samples=20, random_state=42)
    # Convert to binary classification
    binary_labels = (labels >= 2).long()
    return features, binary_labels, ["Positive", "Negative"]


@pytest.fixture
def text_samples():
    """Provide sample text data for testing."""
    return [
        "I am so happy and excited!",
        "This is really sad and disappointing.",
        "I'm feeling angry about this situation.",
        "Just a neutral observation about things.",
        "Another happy moment in life.",
        "Feeling down and melancholy today."
    ]


@pytest.fixture
def text_labels():
    """Provide labels for text samples."""
    return [0, 1, 2, 3, 0, 1]  # Happy, Sad, Angry, Neutral, Happy, Sad


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture(scope="session")
def device():
    """Determine best available device for testing."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def quantum_params():
    """Create sample quantum parameters."""
    return torch.randn(2, 3, 3) * 0.1  # 2 layers, 3 qubits, 3 params each


@pytest.fixture
def model_config():
    """Standard model configuration for testing."""
    return {
        'n_qubits': 4,
        'n_classes': 4,
        'n_layers': 2,
        'feature_dim': 4,
        'hidden_dim': 16
    }


# Test data generators
def generate_separable_data(n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate linearly separable data for testing."""
    torch.manual_seed(42)

    # Create two well-separated clusters
    cluster1 = torch.randn(n_samples // 2, 4) * 0.5 + torch.tensor([2.0, 2.0, 0.0, 0.0])
    cluster2 = torch.randn(n_samples // 2, 4) * 0.5 + torch.tensor([-2.0, -2.0, 0.0, 0.0])

    features = torch.cat([cluster1, cluster2])
    labels = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)]).long()

    # Shuffle
    indices = torch.randperm(n_samples)
    return features[indices], labels[indices]


def generate_noisy_data(n_samples: int = 100, noise_level: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate noisy data for robustness testing."""
    torch.manual_seed(42)

    # Base patterns
    patterns = torch.tensor([
        [0.8, 0.2, 0.7, 0.3],  # Pattern 0
        [0.2, 0.8, 0.3, 0.7],  # Pattern 1
        [0.5, 0.5, 0.9, 0.1],  # Pattern 2
        [0.1, 0.9, 0.5, 0.5],  # Pattern 3
    ])

    features = []
    labels = []

    for i in range(n_samples):
        label = i % 4
        pattern = patterns[label]
        noise = torch.randn(4) * noise_level
        noisy_pattern = torch.clamp(pattern + noise, 0, 1)

        features.append(noisy_pattern)
        labels.append(label)

    return torch.stack(features), torch.tensor(labels)


# Custom test markers
pytest.mark.slow = pytest.mark.mark("slow")
pytest.mark.integration = pytest.mark.mark("integration")
pytest.mark.quantum = pytest.mark.mark("quantum")


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "quantum: marks tests that require quantum computation")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add quantum marker to quantum-related tests
        if "quantum" in item.name.lower() or "circuit" in item.name.lower():
            item.add_marker(pytest.mark.quantum)

        # Add integration marker to end-to-end tests
        if "end_to_end" in item.name.lower() or "classification" in item.name.lower():
            item.add_marker(pytest.mark.integration)

        # Add slow marker to tests that might take longer
        if any(keyword in item.name.lower() for keyword in ["train", "convergence", "optimization"]):
            item.add_marker(pytest.mark.slow)


# Utility functions for tests
def assert_valid_quantum_state(state: torch.Tensor, tolerance: float = 1e-6):
    """Assert that a tensor represents a valid quantum state."""
    assert torch.is_complex(state), "Quantum state must be complex"
    assert state.dim() == 1, "Quantum state must be 1D"

    # Check normalization
    norm_squared = torch.sum(torch.abs(state) ** 2)
    assert torch.abs(norm_squared - 1.0) < tolerance, f"State not normalized: norm² = {norm_squared}"


def assert_valid_probability_distribution(probs: torch.Tensor, tolerance: float = 1e-6):
    """Assert that a tensor represents a valid probability distribution."""
    assert torch.all(probs >= 0), "Probabilities must be non-negative"
    assert torch.all(probs <= 1), "Probabilities must be ≤ 1"

    if probs.dim() == 1:
        # Single distribution
        assert torch.abs(torch.sum(probs) - 1.0) < tolerance, "Probabilities must sum to 1"
    elif probs.dim() == 2:
        # Batch of distributions
        sums = torch.sum(probs, dim=1)
        assert torch.all(torch.abs(sums - 1.0) < tolerance), "Each probability distribution must sum to 1"


def assert_valid_gradients(model: torch.nn.Module, tolerance: float = 1e-8):
    """Assert that model gradients are valid (finite and not all zero)."""
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradients in {name}"

    assert has_grad, "Model has no gradients"


# Test data paths (if needed for file-based tests)
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def test_data_dir():
    """Provide path to test data directory."""
    return TEST_DATA_DIR