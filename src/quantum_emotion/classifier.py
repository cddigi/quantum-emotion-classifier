"""
Main QuantumTextClassifier class for emotion classification.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings


class QuantumTextClassifier(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network for Text Emotion Classification.

    Uses PennyLane for quantum circuits and PyTorch for classical processing.
    Provides exponential feature space scaling with polynomial parameters.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_classes: int = 4,
        n_layers: int = 3,
        backend: str = "default.qubit",
        feature_dim: int = 4,
        hidden_dim: int = 16
    ):
        """
        Initialize hybrid quantum-classical model.

        Args:
            n_qubits: Number of qubits (2^n dimensional Hilbert space)
            n_classes: Number of emotion classes
            n_layers: Number of variational layers in quantum circuit
            backend: PennyLane backend ('default.qubit', 'qiskit.aer', etc.)
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for classical layers
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Create quantum device
        try:
            self.dev = qml.device(backend, wires=n_qubits)
        except Exception as e:
            warnings.warn(f"Backend {backend} not available, using default.qubit: {e}")
            self.dev = qml.device("default.qubit", wires=n_qubits)

        # Define quantum circuit as QNode with parameter-shift gradients
        self.quantum_circuit = qml.QNode(
            self._circuit,
            self.dev,
            interface="torch",
            diff_method="parameter-shift"
        )

        # Initialize trainable quantum parameters
        self.quantum_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3, dtype=torch.float32) * 0.1
        )

        # Classical preprocessing layer
        self.classical_preprocess = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh()  # Bound inputs to [-1, 1] for angle encoding
        )

        # Classical postprocessing layer
        self.classical_postprocess = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes)
        )

        # Store model configuration
        self.config = {
            'n_qubits': n_qubits,
            'n_classes': n_classes,
            'n_layers': n_layers,
            'backend': backend,
            'feature_dim': feature_dim,
            'hidden_dim': hidden_dim,
            'hilbert_dim': 2**n_qubits,
            'quantum_params': n_layers * n_qubits * 3
        }

    def _circuit(self, inputs: torch.Tensor, params: torch.Tensor) -> List[float]:
        """
        Variational quantum circuit with data encoding.

        Args:
            inputs: Classical data to encode (shape: [n_qubits])
            params: Variational parameters (shape: [n_layers, n_qubits, 3])

        Returns:
            List of expectation values for each qubit
        """
        # Validate inputs
        if len(inputs) == 0:
            raise ValueError("Input tensor cannot be empty")

        # Check for NaN values
        if torch.any(torch.isnan(inputs)):
            raise ValueError("Input tensor contains NaN values")

        # Data encoding layer - angle encoding
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            if i < len(inputs):
                qml.RY(inputs[i] * np.pi, wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            # Entangling layer - creates quantum correlations
            for i in range(0, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

            # Parameterized rotations
            for i in range(self.n_qubits):
                qml.RX(params[layer, i, 0], wires=i)
                qml.RY(params[layer, i, 1], wires=i)
                qml.RZ(params[layer, i, 2], wires=i)

            # Additional entanglement for expressivity
            if layer < self.n_layers - 1:
                for i in range(self.n_qubits):
                    qml.CRZ(
                        params[layer, i, 0],
                        wires=[i, (i + 1) % self.n_qubits]
                    )

        # Measurement - Pauli-Z expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid network.

        Args:
            x: Input tensor of shape (batch_size, feature_dim)

        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        batch_size = x.shape[0]

        # Classical preprocessing
        x_preprocessed = self.classical_preprocess(x)

        # Process each sample through quantum circuit
        quantum_outputs = []
        for i in range(batch_size):
            q_out = self.quantum_circuit(x_preprocessed[i], self.quantum_params)
            # Convert quantum output to tensor while preserving gradients
            if isinstance(q_out, (list, tuple)):
                # Convert to tensor but preserve gradient flow
                q_out_tensor = torch.stack([torch.as_tensor(val, dtype=torch.float32, device=x.device) for val in q_out])
            else:
                q_out_tensor = q_out
            quantum_outputs.append(q_out_tensor)

        # Stack quantum outputs
        quantum_tensor = torch.stack(quantum_outputs)

        # Classical postprocessing
        output = self.classical_postprocess(quantum_tensor)

        return output

    def get_quantum_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the quantum state vector for analysis.

        Args:
            x: Input tensor of shape (feature_dim,)

        Returns:
            Complex quantum state vector
        """
        @qml.qnode(self.dev)
        def state_circuit(inputs, params):
            # Same circuit but return state
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                if i < len(inputs):
                    qml.RY(inputs[i] * np.pi, wires=i)

            for layer in range(self.n_layers):
                for i in range(0, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])

                for i in range(self.n_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                    qml.RZ(params[layer, i, 2], wires=i)

                if layer < self.n_layers - 1:
                    for i in range(self.n_qubits):
                        qml.CRZ(
                            params[layer, i, 0],
                            wires=[i, (i + 1) % self.n_qubits]
                        )

            return qml.state()

        x_processed = self.classical_preprocess(x.unsqueeze(0))[0]
        return state_circuit(x_processed, self.quantum_params)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model configuration and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        quantum_params = self.quantum_params.numel()
        classical_params = total_params - quantum_params

        return {
            **self.config,
            'total_parameters': total_params,
            'quantum_parameters': quantum_params,
            'classical_parameters': classical_params,
            'compression_ratio': self.config['hilbert_dim'] / quantum_params
        }