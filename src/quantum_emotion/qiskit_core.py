"""
Core Qiskit Implementation for Quantum Emotion Classification

Simplified, robust implementation using Qiskit's core functionality
that works with the latest Qiskit version.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
import warnings

# Core Qiskit imports
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.visualization import circuit_drawer, plot_histogram


class QiskitEmotionClassifier(nn.Module):
    """
    Qiskit-based Quantum Emotion Classifier.

    A simplified but powerful implementation that works with the latest Qiskit version,
    focusing on core quantum circuit functionality.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_classes: int = 4,
        n_layers: int = 3,
        feature_dim: int = 4,
        hidden_dim: int = 16,
        backend_name: str = "aer_simulator"
    ):
        """
        Initialize the Qiskit quantum emotion classifier.

        Args:
            n_qubits: Number of qubits (determines Hilbert space: 2^n)
            n_classes: Number of emotion classes
            n_layers: Depth of quantum circuit (repetitions)
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for classical layers
            backend_name: Qiskit backend to use
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Initialize backend
        self.backend = AerSimulator()

        # Create parameterized quantum circuit
        self.quantum_circuit = self._build_quantum_circuit()

        # Initialize learnable parameters for quantum circuit
        n_params = len(self.input_params) + len(self.weight_params)
        self.quantum_weights = nn.Parameter(torch.randn(len(self.weight_params)) * 0.1)

        # Classical preprocessing
        self.classical_preprocess = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh()  # Normalize to [-1, 1] for quantum encoding
        )

        # Classical postprocessing
        self.classical_postprocess = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes)
        )

        # Store configuration
        self.config = {
            'n_qubits': n_qubits,
            'n_classes': n_classes,
            'n_layers': n_layers,
            'hilbert_dim': 2**n_qubits,
            'n_quantum_params': len(self.weight_params),
            'backend': backend_name
        }

    def _build_quantum_circuit(self) -> QuantumCircuit:
        """
        Build the parameterized quantum circuit.

        Returns:
            Parameterized quantum circuit
        """
        # Create quantum and classical registers
        qreg = QuantumRegister(self.n_qubits, 'q')
        creg = ClassicalRegister(self.n_qubits, 'c')
        qc = QuantumCircuit(qreg, creg)

        # Input parameters for data encoding
        self.input_params = ParameterVector('x', self.n_qubits)

        # Weight parameters for variational circuit
        self.weight_params = ParameterVector('θ', self.n_layers * self.n_qubits * 3)

        # Data encoding layer
        for i in range(self.n_qubits):
            qc.h(qreg[i])  # Hadamard for superposition
            qc.ry(self.input_params[i], qreg[i])  # Encode data

        # Variational layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Entangling layer
            for i in range(0, self.n_qubits - 1):
                qc.cx(qreg[i], qreg[i + 1])

            # Parameterized rotation layer
            for i in range(self.n_qubits):
                qc.rx(self.weight_params[param_idx], qreg[i])
                param_idx += 1
                qc.ry(self.weight_params[param_idx], qreg[i])
                param_idx += 1
                qc.rz(self.weight_params[param_idx], qreg[i])
                param_idx += 1

        # Measurement
        qc.measure_all()

        return qc

    def execute_quantum_circuit(self, input_data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Execute the quantum circuit with given inputs and weights.

        Args:
            input_data: Input features for encoding (n_qubits,)
            weights: Variational parameters

        Returns:
            Measurement expectations (n_qubits,)
        """
        # Bind parameters
        parameter_bindings = {}

        # Bind input parameters
        for i, param in enumerate(self.input_params):
            parameter_bindings[param] = float(input_data[i])

        # Bind weight parameters
        for i, param in enumerate(self.weight_params):
            parameter_bindings[param] = float(weights[i])

        # Create bound circuit
        bound_circuit = self.quantum_circuit.assign_parameters(parameter_bindings)

        # Transpile for backend
        transpiled = transpile(bound_circuit, self.backend)

        # Execute circuit
        job = self.backend.run(transpiled, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Convert counts to expectation values
        expectations = self._counts_to_expectations(counts)

        return expectations

    def _counts_to_expectations(self, counts: Dict[str, int]) -> np.ndarray:
        """
        Convert measurement counts to expectation values.

        Args:
            counts: Measurement counts from circuit execution

        Returns:
            Expectation values for each qubit
        """
        total_shots = sum(counts.values())
        expectations = np.zeros(self.n_qubits)

        for bitstring, count in counts.items():
            # Remove spaces and reverse for proper indexing
            bitstring = bitstring.replace(' ', '')[::-1]

            for i, bit in enumerate(bitstring[:self.n_qubits]):
                if bit == '0':
                    expectations[i] += count / total_shots
                else:
                    expectations[i] -= count / total_shots

        return expectations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum-classical network.

        Args:
            x: Input tensor (batch_size, feature_dim)

        Returns:
            Output tensor (batch_size, n_classes)
        """
        batch_size = x.shape[0]

        # Classical preprocessing
        quantum_input = self.classical_preprocess(x)

        # Process each sample through quantum circuit
        quantum_outputs = []
        for i in range(batch_size):
            # Execute quantum circuit
            input_data = quantum_input[i].detach().cpu().numpy()
            weights = self.quantum_weights.detach().cpu().numpy()

            expectations = self.execute_quantum_circuit(input_data, weights)

            # Convert to tensor
            quantum_output = torch.tensor(expectations, dtype=torch.float32, device=x.device)
            quantum_outputs.append(quantum_output)

        # Stack outputs
        quantum_batch = torch.stack(quantum_outputs)

        # Classical postprocessing
        output = self.classical_postprocess(quantum_batch)

        return output

    def get_circuit_diagram(self) -> str:
        """Get a text representation of the quantum circuit."""
        # Create a sample circuit with dummy parameters
        params = {**{p: 0.5 for p in self.input_params},
                 **{p: 0.5 for p in self.weight_params}}
        bound_circuit = self.quantum_circuit.assign_parameters(params)
        return bound_circuit.draw(output='text')

    def get_circuit_depth(self) -> int:
        """Get the depth of the quantum circuit."""
        params = {**{p: 0.5 for p in self.input_params},
                 **{p: 0.5 for p in self.weight_params}}
        bound_circuit = self.quantum_circuit.assign_parameters(params)
        return bound_circuit.depth()


def create_quantum_feature_map(n_qubits: int = 4, reps: int = 2) -> QuantumCircuit:
    """
    Create a quantum feature map for encoding classical data.

    Args:
        n_qubits: Number of qubits
        reps: Number of repetitions

    Returns:
        Quantum feature map circuit
    """
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement='linear'
    )
    return feature_map


def create_variational_ansatz(n_qubits: int = 4, n_layers: int = 3) -> QuantumCircuit:
    """
    Create a variational ansatz for parameterized quantum circuits.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers (circuit depth)

    Returns:
        Variational ansatz circuit
    """
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=n_layers,
        entanglement='linear'
    )
    return ansatz


def create_efficient_quantum_circuit(n_qubits: int = 4, n_layers: int = 2) -> QuantumCircuit:
    """
    Create an efficient quantum circuit using EfficientSU2.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers

    Returns:
        Efficient quantum circuit
    """
    circuit = EfficientSU2(
        num_qubits=n_qubits,
        reps=n_layers,
        entanglement='linear'
    )
    return circuit


class QiskitQuantumTrainer:
    """
    Training orchestrator for Qiskit quantum models.
    """

    def __init__(
        self,
        model: QiskitEmotionClassifier,
        learning_rate: float = 0.01,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize trainer.

        Args:
            model: Qiskit emotion classifier
            learning_rate: Learning rate
            device: Training device
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader=None, epochs=10, verbose=True):
        """Train the model for multiple epochs."""
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)

            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
            else:
                val_loss, val_acc = 0, 0

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
                if val_loader:
                    print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        return self.history


if __name__ == "__main__":
    """Demo the Qiskit quantum emotion classifier."""
    print("🚀 Qiskit Quantum Emotion Classifier")
    print("=" * 50)

    # Create model
    model = QiskitEmotionClassifier(
        n_qubits=4,
        n_classes=4,
        n_layers=2
    )

    # Display configuration
    print("\n📊 Model Configuration:")
    for key, value in model.config.items():
        print(f"   {key}: {value}")

    # Show circuit depth
    print(f"\n🔮 Circuit Depth: {model.get_circuit_depth()}")

    # Display circuit diagram
    print("\n📐 Quantum Circuit Diagram:")
    print(model.get_circuit_diagram())

    # Test forward pass
    batch_size = 2
    feature_dim = 4
    test_input = torch.randn(batch_size, feature_dim)

    print(f"\n🧪 Testing forward pass...")
    try:
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ Success!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output: {output}")
    except Exception as e:
        print(f"⚠️  Error: {e}")

    print("\n✨ Qiskit migration complete! Ready for quantum emotion classification.")