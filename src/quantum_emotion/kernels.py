"""
Quantum kernel methods for similarity computation and SVM-style classification.
"""

import pennylane as qml
import torch
import numpy as np
from typing import List, Tuple, Optional, Callable
from sklearn.svm import SVC
import warnings


class QuantumKernel:
    """
    Quantum kernel methods for computing similarities in exponential feature space.

    Provides quantum kernel functions that operate in 2^n dimensional Hilbert space
    using only n qubits, enabling exponential speedup for certain kernel computations.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        backend: str = "default.qubit",
        encoding_type: str = "angle"
    ):
        """
        Initialize quantum kernel computer.

        Args:
            n_qubits: Number of qubits
            backend: PennyLane backend
            encoding_type: Type of data encoding ('angle', 'amplitude', 'iqp')
        """
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type

        try:
            self.dev = qml.device(backend, wires=n_qubits)
        except Exception as e:
            warnings.warn(f"Backend {backend} not available, using default.qubit: {e}")
            self.dev = qml.device("default.qubit", wires=n_qubits)

        # Create kernel circuit based on encoding type
        if encoding_type == "angle":
            self.kernel_circuit = self._create_angle_kernel()
        elif encoding_type == "amplitude":
            self.kernel_circuit = self._create_amplitude_kernel()
        elif encoding_type == "iqp":
            self.kernel_circuit = self._create_iqp_kernel()
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

    def _create_angle_kernel(self) -> Callable:
        """Create angle encoding quantum kernel."""
        @qml.qnode(self.dev)
        def angle_kernel(x1: np.ndarray, x2: np.ndarray):
            # Simple feature map approach - compute inner product of feature maps
            # |phi(x1)><phi(x2)| = Tr(rho1 * rho2) where rho = |phi><phi|

            # Feature map for first input
            for i in range(min(len(x1), self.n_qubits)):
                qml.RY(x1[i] * np.pi, wires=i)

            # Add entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Adjoint of feature map for second input
            for i in range(self.n_qubits - 1, 0, -1):
                qml.CNOT(wires=[i - 1, i])

            for i in range(min(len(x2), self.n_qubits) - 1, -1, -1):
                qml.RY(-x2[i] * np.pi, wires=i)

            # Measure fidelity as probability of |0...0>
            return qml.probs(wires=range(self.n_qubits))

        return angle_kernel

    def _create_amplitude_kernel(self) -> Callable:
        """Create amplitude encoding quantum kernel."""
        @qml.qnode(self.dev)
        def amplitude_kernel(x1: np.ndarray, x2: np.ndarray):
            # Normalize inputs
            x1_norm = x1 / (np.linalg.norm(x1) + 1e-8)
            x2_norm = x2 / (np.linalg.norm(x2) + 1e-8)

            # Pad to 2^n_qubits dimension
            dim = 2 ** self.n_qubits
            x1_padded = np.zeros(dim)
            x2_padded = np.zeros(dim)

            x1_padded[:min(len(x1_norm), dim)] = x1_norm[:min(len(x1_norm), dim)]
            x2_padded[:min(len(x2_norm), dim)] = x2_norm[:min(len(x2_norm), dim)]

            # Encode first sample
            qml.QubitStateVector(x1_padded, wires=range(self.n_qubits))

            # Apply adjoint of second sample preparation
            qml.adjoint(qml.QubitStateVector)(x2_padded, wires=range(self.n_qubits))

            # Return probability of all zeros state
            return qml.probs(wires=range(self.n_qubits))

        return amplitude_kernel

    def _create_iqp_kernel(self) -> Callable:
        """Create Instantaneous Quantum Polynomial (IQP) kernel."""
        @qml.qnode(self.dev)
        def iqp_kernel(x1: np.ndarray, x2: np.ndarray):
            # Initialize superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # First layer: encode x1
            for i in range(min(len(x1), self.n_qubits)):
                qml.RZ(2 * x1[i], wires=i)

            # Interaction layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                if i < len(x1) - 1:
                    qml.RZ(2 * x1[i] * x1[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])

            # Second layer: encode x2 (inverse)
            for i in range(min(len(x2), self.n_qubits)):
                qml.RZ(-2 * x2[i], wires=i)

            # Final superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # Return probability of all zeros state
            return qml.probs(wires=range(self.n_qubits))

        return iqp_kernel

    def compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute quantum kernel between two samples.

        Args:
            x1: First sample
            x2: Second sample

        Returns:
            Kernel value between 0 and 1
        """
        probs = self.kernel_circuit(x1, x2)
        return float(probs[0])

    def kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute full kernel matrix for dataset.

        Args:
            X: Dataset of shape (n_samples, n_features)

        Returns:
            Kernel matrix of shape (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):
                kernel_val = self.compute_kernel(X[i], X[j])
                K[i, j] = kernel_val
                K[j, i] = kernel_val  # Symmetric

        return K

    def kernel_alignment(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute kernel-target alignment metric.

        Args:
            X: Features
            y: Labels

        Returns:
            Alignment score between -1 and 1
        """
        K = self.kernel_matrix(X)
        n = len(y)

        # Create ideal kernel matrix
        Y = np.outer(y, y)
        ideal_K = (Y == Y.T).astype(float)

        # Compute alignment
        numerator = np.trace(K @ ideal_K)
        denominator = np.sqrt(np.trace(K @ K) * np.trace(ideal_K @ ideal_K))

        return numerator / (denominator + 1e-8)


class QuantumSVM:
    """
    Quantum Support Vector Machine using quantum kernels.

    Combines quantum kernel computation with classical SVM optimization
    for hybrid quantum-classical classification.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        backend: str = "default.qubit",
        encoding_type: str = "angle",
        C: float = 1.0,
        **svm_kwargs
    ):
        """
        Initialize Quantum SVM.

        Args:
            n_qubits: Number of qubits for quantum kernel
            backend: PennyLane backend
            encoding_type: Quantum encoding type
            C: SVM regularization parameter
            **svm_kwargs: Additional arguments for sklearn SVM
        """
        self.quantum_kernel = QuantumKernel(n_qubits, backend, encoding_type)
        self.svm = SVC(kernel='precomputed', C=C, **svm_kwargs)
        self.X_train = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantumSVM':
        """
        Fit quantum SVM to training data.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        self.X_train = X.copy()

        print(f"Computing quantum kernel matrix for {len(X)} samples...")
        K_train = self.quantum_kernel.kernel_matrix(X)

        print("Training SVM with quantum kernel...")
        self.svm.fit(K_train, y)
        self.is_fitted = True

        # Compute kernel alignment
        alignment = self.quantum_kernel.kernel_alignment(X, y)
        print(f"Kernel-target alignment: {alignment:.3f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for new samples.

        Args:
            X: Test features

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        print(f"Computing kernel matrix for {len(X)} test samples...")
        # Compute kernel between test and training samples
        K_test = np.zeros((X.shape[0], self.X_train.shape[0]))

        for i in range(X.shape[0]):
            for j in range(self.X_train.shape[0]):
                K_test[i, j] = self.quantum_kernel.compute_kernel(X[i], self.X_train[j])

        return self.svm.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for new samples.

        Args:
            X: Test features

        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Compute kernel matrix
        K_test = np.zeros((X.shape[0], self.X_train.shape[0]))

        for i in range(X.shape[0]):
            for j in range(self.X_train.shape[0]):
                K_test[i, j] = self.quantum_kernel.compute_kernel(X[i], self.X_train[j])

        return self.svm.predict_proba(K_test)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy score on test data.

        Args:
            X: Test features
            y: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def compare_kernels(X: np.ndarray, y: np.ndarray, n_qubits: int = 4) -> dict:
    """
    Compare different quantum kernel encodings.

    Args:
        X: Dataset features
        y: Dataset labels
        n_qubits: Number of qubits

    Returns:
        Dictionary with alignment scores for each encoding
    """
    encodings = ["angle", "iqp"]  # Skip amplitude for simplicity
    results = {}

    for encoding in encodings:
        print(f"\nTesting {encoding} encoding...")
        try:
            kernel = QuantumKernel(n_qubits=n_qubits, encoding_type=encoding)
            alignment = kernel.kernel_alignment(X, y)
            results[encoding] = alignment
            print(f"{encoding} kernel alignment: {alignment:.3f}")
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
            results[encoding] = float('nan')

    return results