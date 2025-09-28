"""
Qiskit-based Quantum Text Classifier for Emotion Classification

Complete migration from PennyLane to Qiskit for quantum machine learning,
leveraging IBM's quantum computing framework and ecosystem.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Dict, Any, Union
import warnings

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal, EfficientSU2
from qiskit_machine_learning.kernels import QuantumKernel, FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, VQC, NeuralNetworkClassifier
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class QiskitQuantumTextClassifier(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network for Text Emotion Classification
    using IBM Qiskit framework.

    Provides exponential feature space scaling with polynomial parameters
    through quantum circuit processing.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_classes: int = 4,
        n_layers: int = 3,
        feature_dim: int = 4,
        hidden_dim: int = 16,
        backend: str = "aer_simulator",
        shots: int = 1024,
        use_quantum_kernel: bool = False,
        optimization_level: int = 3
    ):
        """
        Initialize Qiskit-based quantum classifier.

        Args:
            n_qubits: Number of qubits (2^n dimensional Hilbert space)
            n_classes: Number of emotion classes
            n_layers: Number of variational layers (circuit depth)
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for classical layers
            backend: Qiskit backend ('aer_simulator', 'ibmq_qasm_simulator', etc.)
            shots: Number of measurement shots
            use_quantum_kernel: Whether to use quantum kernel methods
            optimization_level: Qiskit transpiler optimization level (0-3)
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.shots = shots
        self.use_quantum_kernel = use_quantum_kernel
        self.optimization_level = optimization_level

        # Initialize quantum backend
        self.backend = self._initialize_backend(backend)

        # Create quantum circuit components
        self.feature_map = self._create_feature_map()
        self.ansatz = self._create_ansatz()
        self.quantum_circuit = self._build_quantum_circuit()

        # Initialize quantum neural network
        if use_quantum_kernel:
            self.quantum_kernel = self._create_quantum_kernel()
        else:
            self.quantum_nn = self._create_quantum_nn()
            # Create PyTorch connector for gradient-based training
            self.torch_connector = TorchConnector(self.quantum_nn)

        # Classical preprocessing layers
        self.classical_preprocess = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh()  # Bound inputs to [-1, 1] for angle encoding
        )

        # Classical postprocessing layers
        self.classical_postprocess = nn.Sequential(
            nn.Linear(n_qubits if not use_quantum_kernel else hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes)
        )

        # Store configuration
        self.config = {
            'n_qubits': n_qubits,
            'n_classes': n_classes,
            'n_layers': n_layers,
            'backend': backend,
            'feature_dim': feature_dim,
            'hidden_dim': hidden_dim,
            'hilbert_dim': 2**n_qubits,
            'quantum_params': self.ansatz.num_parameters if hasattr(self, 'ansatz') else n_layers * n_qubits * 3,
            'shots': shots,
            'use_quantum_kernel': use_quantum_kernel
        }

    def _initialize_backend(self, backend_name: str):
        """Initialize Qiskit backend."""
        if backend_name == "aer_simulator":
            return AerSimulator()
        elif backend_name == "statevector_simulator":
            return AerSimulator(method='statevector')
        elif backend_name == "gpu_simulator":
            # GPU acceleration if available
            try:
                return AerSimulator(method='statevector', device='GPU')
            except:
                warnings.warn("GPU not available, falling back to CPU")
                return AerSimulator(method='statevector')
        else:
            # For IBM Quantum backends, would need authentication
            warnings.warn(f"Backend {backend_name} requires IBM Quantum account setup")
            return AerSimulator()

    def _create_feature_map(self) -> QuantumCircuit:
        """
        Create quantum feature map for encoding classical data.

        Uses ZZFeatureMap for efficient encoding with entanglement.
        """
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=2,
            entanglement='linear',
            parameter_prefix='x'
        )
        return feature_map

    def _create_ansatz(self) -> QuantumCircuit:
        """
        Create variational ansatz (parameterized quantum circuit).

        Uses RealAmplitudes for hardware-efficient design.
        """
        ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=self.n_layers,
            entanglement='linear',
            parameter_prefix='θ'
        )
        return ansatz

    def _build_quantum_circuit(self) -> QuantumCircuit:
        """
        Build complete quantum circuit combining feature map and ansatz.
        """
        qc = QuantumCircuit(self.n_qubits)

        # Combine feature map and ansatz
        qc.compose(self.feature_map, inplace=True)
        qc.compose(self.ansatz, inplace=True)

        return qc

    def _create_quantum_nn(self) -> Union[SamplerQNN, EstimatorQNN]:
        """
        Create quantum neural network using Qiskit ML.

        Returns EstimatorQNN for expectation value-based training.
        """
        from qiskit.quantum_info import SparsePauliOp

        # Create observables for measurement
        observables = []
        for i in range(self.n_qubits):
            # Measure Z on each qubit
            pauli_string = 'I' * i + 'Z' + 'I' * (self.n_qubits - i - 1)
            observables.append(SparsePauliOp.from_list([(pauli_string, 1.0)]))

        # Create EstimatorQNN for gradient-based optimization
        qnn = EstimatorQNN(
            circuit=self.quantum_circuit,
            observables=observables,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            gradient=True  # Enable gradient computation
        )

        return qnn

    def _create_quantum_kernel(self) -> FidelityQuantumKernel:
        """
        Create quantum kernel for kernel-based methods.
        """
        from qiskit_algorithms.state_fidelities import ComputeUncompute

        kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=ComputeUncompute(sampler=Sampler())
        )

        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum-classical hybrid network.

        Args:
            x: Input tensor (batch_size, feature_dim)

        Returns:
            Output tensor with class predictions (batch_size, n_classes)
        """
        batch_size = x.shape[0]

        # Classical preprocessing
        classical_features = self.classical_preprocess(x)

        if self.use_quantum_kernel:
            # Kernel-based approach
            # Note: Kernel methods are typically used differently
            # This is a simplified integration for demonstration
            quantum_output = classical_features  # Placeholder
        else:
            # Neural network approach with Torch connector
            # Ensure features are properly shaped for quantum circuit
            quantum_input = classical_features.view(batch_size, self.n_qubits)

            # Process through quantum circuit via Torch connector
            quantum_output = self.torch_connector(quantum_input)

            # Reshape if needed
            if len(quantum_output.shape) == 1:
                quantum_output = quantum_output.view(batch_size, -1)

        # Classical postprocessing
        output = self.classical_postprocess(quantum_output)

        return output

    def get_circuit_depth(self) -> int:
        """Get the depth of the quantum circuit."""
        return self.quantum_circuit.depth()

    def get_circuit_stats(self) -> Dict[str, Any]:
        """Get statistics about the quantum circuit."""
        return {
            'n_qubits': self.n_qubits,
            'circuit_depth': self.quantum_circuit.depth(),
            'n_parameters': self.ansatz.num_parameters,
            'n_gates': len(self.quantum_circuit.data),
            'feature_map_params': len(self.feature_map.parameters),
            'ansatz_params': len(self.ansatz.parameters)
        }

    def visualize_circuit(self, output_format: str = 'mpl'):
        """
        Visualize the quantum circuit.

        Args:
            output_format: 'mpl' for matplotlib, 'text' for ASCII
        """
        return self.quantum_circuit.draw(output=output_format)


class QiskitQuantumKernel:
    """
    Quantum kernel methods using Qiskit for emotion classification.

    Implements quantum kernel-based SVM and other kernel methods.
    """

    def __init__(self, n_qubits: int = 4, feature_map_reps: int = 2):
        """
        Initialize quantum kernel.

        Args:
            n_qubits: Number of qubits
            feature_map_reps: Number of repetitions in feature map
        """
        self.n_qubits = n_qubits
        self.feature_map_reps = feature_map_reps

        # Create feature map
        self.feature_map = ZZFeatureMap(
            feature_dimension=n_qubits,
            reps=feature_map_reps,
            entanglement='full'
        )

        # Create quantum kernel
        from qiskit_algorithms.state_fidelities import ComputeUncompute

        self.quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=ComputeUncompute(sampler=Sampler())
        )

    def compute_kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute quantum kernel matrix.

        Args:
            X: First dataset (n_samples, n_features)
            Y: Second dataset (optional, defaults to X)

        Returns:
            Kernel matrix
        """
        # Ensure features match qubit count
        if X.shape[1] != self.n_qubits:
            # Pad or truncate features
            X_processed = np.zeros((X.shape[0], self.n_qubits))
            min_features = min(X.shape[1], self.n_qubits)
            X_processed[:, :min_features] = X[:, :min_features]
            X = X_processed

        if Y is not None and Y.shape[1] != self.n_qubits:
            Y_processed = np.zeros((Y.shape[0], self.n_qubits))
            min_features = min(Y.shape[1], self.n_qubits)
            Y_processed[:, :min_features] = Y[:, :min_features]
            Y = Y_processed

        # Compute kernel matrix
        kernel_matrix = self.quantum_kernel.evaluate(x_vec=X, y_vec=Y)

        return kernel_matrix

    def train_qsvc(self, X_train: np.ndarray, y_train: np.ndarray) -> QSVC:
        """
        Train Quantum Support Vector Classifier.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained QSVC model
        """
        qsvc = QSVC(quantum_kernel=self.quantum_kernel)
        qsvc.fit(X_train, y_train)
        return qsvc


class QiskitHybridTrainer:
    """
    Training orchestrator for Qiskit quantum-classical hybrid models.
    """

    def __init__(
        self,
        model: QiskitQuantumTextClassifier,
        learning_rate: float = 0.01,
        optimizer_type: str = 'adam',
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize hybrid trainer.

        Args:
            model: Qiskit quantum text classifier
            learning_rate: Learning rate for optimization
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            device: Training device (CPU/GPU)
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        # Initialize optimizer
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
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

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            verbose: Whether to print progress

        Returns:
            Training history
        """
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
            else:
                val_loss, val_acc = 0.0, 0.0

            # Print progress
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                if val_loader is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        return self.history


def create_qiskit_vqc_classifier(n_qubits: int = 4, n_classes: int = 4) -> VQC:
    """
    Create a Variational Quantum Classifier using Qiskit.

    Args:
        n_qubits: Number of qubits
        n_classes: Number of output classes

    Returns:
        VQC model
    """
    # Create feature map and ansatz
    feature_map = ZZFeatureMap(n_qubits)
    ansatz = EfficientSU2(n_qubits, reps=3)

    # Create VQC
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=100),
        quantum_instance=AerSimulator()
    )

    return vqc


if __name__ == "__main__":
    """Demo Qiskit quantum classifier."""
    print("🎉 Qiskit Quantum Text Classifier Initialized!")
    print("=" * 50)

    # Create model
    model = QiskitQuantumTextClassifier(
        n_qubits=4,
        n_classes=4,
        n_layers=2,
        backend="aer_simulator"
    )

    # Print circuit statistics
    stats = model.get_circuit_stats()
    print(f"📊 Circuit Statistics:")
    print(f"   Qubits: {stats['n_qubits']}")
    print(f"   Depth: {stats['circuit_depth']}")
    print(f"   Parameters: {stats['n_parameters']}")
    print(f"   Gates: {stats['n_gates']}")

    # Visualize circuit (text format for demo)
    print(f"\n🔮 Quantum Circuit:")
    print(model.visualize_circuit(output_format='text'))

    # Test forward pass
    batch_size = 4
    feature_dim = 4
    test_input = torch.randn(batch_size, feature_dim)

    try:
        output = model(test_input)
        print(f"\n✅ Forward pass successful!")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
    except Exception as e:
        print(f"\n⚠️  Forward pass needs full setup: {e}")

    print(f"\n🚀 Ready for Qiskit-based quantum emotion classification!")