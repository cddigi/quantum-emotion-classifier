"""
Hybrid quantum-classical models and training utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from .classifier import QuantumTextClassifier


class HybridModel(nn.Module):
    """
    PyTorch wrapper for quantum circuits with automatic differentiation.

    Provides a standard PyTorch interface for hybrid quantum-classical models
    with additional quantum-specific functionality.
    """

    def __init__(self, quantum_classifier: QuantumTextClassifier):
        """
        Initialize hybrid model wrapper.

        Args:
            quantum_classifier: QuantumTextClassifier instance
        """
        super().__init__()
        self.quantum_classifier = quantum_classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid model."""
        return self.quantum_classifier(x)

    def get_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract quantum features before classical postprocessing."""
        batch_size = x.shape[0]
        x_preprocessed = self.quantum_classifier.classical_preprocess(x)

        quantum_outputs = []
        for i in range(batch_size):
            q_out = self.quantum_classifier.quantum_circuit(
                x_preprocessed[i],
                self.quantum_classifier.quantum_params
            )
            quantum_outputs.append(q_out)

        return torch.stack(quantum_outputs)

    def quantum_feature_analysis(self, x: torch.Tensor) -> Dict[str, Any]:
        """Analyze quantum features for interpretability."""
        quantum_features = self.get_quantum_features(x)

        return {
            'mean_features': quantum_features.mean(dim=0),
            'std_features': quantum_features.std(dim=0),
            'feature_range': {
                'min': quantum_features.min(dim=0)[0],
                'max': quantum_features.max(dim=0)[0]
            },
            'feature_correlations': torch.corrcoef(quantum_features.T)
        }


class HybridTrainer:
    """
    Training framework for hybrid quantum-classical models.

    Handles quantum gradient computation, training loops, and evaluation
    with quantum-specific metrics and visualizations.
    """

    def __init__(
        self,
        model: QuantumTextClassifier,
        learning_rate: float = 0.01,
        device: Optional[torch.device] = None
    ):
        """
        Initialize hybrid trainer.

        Args:
            model: Quantum classifier model
            learning_rate: Learning rate for optimization
            device: Device for computation (CPU/MPS/CUDA)
        """
        self.model = model
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'quantum_grad_norms': [],
            'classical_grad_norms': [],
            'learning_rates': []
        }

        # Best model state
        self.best_val_accuracy = 0.0
        self.best_model_state = None

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
        quantum_grad_norms = []
        classical_grad_norms = []

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Analyze gradients
            if self.model.quantum_params.grad is not None:
                q_grad_norm = torch.norm(self.model.quantum_params.grad).item()
                quantum_grad_norms.append(q_grad_norm)

            # Classical gradient norms
            classical_grads = []
            for name, param in self.model.named_parameters():
                if 'quantum_params' not in name and param.grad is not None:
                    classical_grads.append(torch.norm(param.grad).item())

            if classical_grads:
                classical_grad_norms.append(np.mean(classical_grads))

            # Update parameters
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Update history
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        self.history['train_loss'].append(avg_loss)
        self.history['train_accuracy'].append(accuracy)

        if quantum_grad_norms:
            self.history['quantum_grad_norms'].extend(quantum_grad_norms)
        if classical_grad_norms:
            self.history['classical_grad_norms'].extend(classical_grad_norms)

        return avg_loss, accuracy

    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on validation set.

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

        self.history['val_loss'].append(avg_loss)
        self.history['val_accuracy'].append(accuracy)

        # Save best model
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
            self.best_model_state = self.model.state_dict().copy()

        return avg_loss, accuracy

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        if verbose:
            print(f"Training hybrid quantum-classical model for {epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"Model info: {self.model.get_model_info()}")

        patience_counter = 0
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and epoch % 5 == 0:
                    print(f"Epoch {epoch:3d} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

                # Early stopping
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            else:
                if verbose and epoch % 5 == 0:
                    print(f"Epoch {epoch:3d} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")

        if verbose:
            print("Training completed!")
            if val_loader is not None:
                print(f"Best validation accuracy: {self.best_val_accuracy:.1f}%")

        return self.history

    def detailed_evaluation(
        self,
        test_loader: torch.utils.data.DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation with detailed metrics.

        Args:
            test_loader: Test data loader
            class_names: Names of classes for reporting

        Returns:
            Dictionary with detailed evaluation results
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        accuracy = np.mean(all_predictions == all_targets)

        # Classification report
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(all_targets)))]

        report = classification_report(
            all_targets, all_predictions,
            target_names=class_names,
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)

        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names
        }

    def plot_training_history(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot comprehensive training history."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val Loss', color='red')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(self.history['train_accuracy'], label='Train Acc', color='blue')
        if self.history['val_accuracy']:
            axes[1].plot(self.history['val_accuracy'], label='Val Acc', color='red')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)

        # Gradient norms
        if self.history['quantum_grad_norms']:
            window = min(50, len(self.history['quantum_grad_norms']) // 10 + 1)
            q_grads_smooth = np.convolve(
                self.history['quantum_grad_norms'],
                np.ones(window) / window,
                mode='valid'
            )
            axes[2].plot(q_grads_smooth, label='Quantum Grads', color='green')

        if self.history['classical_grad_norms']:
            window = min(50, len(self.history['classical_grad_norms']) // 10 + 1)
            c_grads_smooth = np.convolve(
                self.history['classical_grad_norms'],
                np.ones(window) / window,
                mode='valid'
            )
            axes[2].plot(c_grads_smooth, label='Classical Grads', color='orange')

        axes[2].set_title('Gradient Norms (Smoothed)')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Gradient Norm')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_yscale('log')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, evaluation_results: Dict[str, Any]) -> None:
        """Plot confusion matrix from evaluation results."""
        cm = evaluation_results['confusion_matrix']
        class_names = evaluation_results['class_names']

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def load_best_model(self) -> None:
        """Load the best model state from training."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        else:
            print("No best model state available")