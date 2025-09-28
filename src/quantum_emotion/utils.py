"""
Utility functions for quantum emotion classification.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pennylane as qml


def create_emotion_dataset(
    n_samples: int = 1000,
    noise_level: float = 0.1,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Create synthetic emotion dataset for testing.

    Args:
        n_samples: Number of samples to generate
        noise_level: Amount of noise to add
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features, labels, emotion_names)
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    emotion_names = ["Happy", "Sad", "Angry", "Neutral"]
    n_classes = len(emotion_names)
    samples_per_class = n_samples // n_classes

    # Define emotion prototypes in feature space
    prototypes = {
        0: [0.8, 0.2, 0.7, 0.3],   # Happy: high positive, low negative
        1: [0.2, 0.8, 0.3, 0.7],   # Sad: low positive, high negative
        2: [0.3, 0.7, 0.9, 0.1],   # Angry: medium positive, high intensity
        3: [0.5, 0.5, 0.4, 0.6]    # Neutral: balanced features
    }

    features = []
    labels = []

    for class_idx in range(n_classes):
        prototype = np.array(prototypes[class_idx])

        for _ in range(samples_per_class):
            # Add Gaussian noise around prototype
            sample = prototype + np.random.normal(0, noise_level, len(prototype))
            sample = np.clip(sample, 0, 1)  # Keep in valid range

            features.append(sample)
            labels.append(class_idx)

    # Convert to tensors - convert to numpy first for performance
    features = torch.tensor(np.array(features), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)

    # Shuffle the dataset
    indices = torch.randperm(len(features))
    features = features[indices]
    labels = labels[indices]

    return features, labels, emotion_names


def create_text_emotion_dataset() -> Tuple[List[str], List[int], List[str]]:
    """
    Create realistic text emotion dataset.

    Returns:
        Tuple of (texts, labels, emotion_names)
    """
    emotion_names = ["Happy", "Sad", "Angry", "Neutral"]

    texts = [
        # Happy (0)
        "I am so excited about this amazing opportunity!",
        "What a beautiful sunny day, feeling fantastic!",
        "Just got great news, couldn't be happier!",
        "Love spending time with friends and family.",
        "This is the best day ever, everything is perfect!",
        "Feeling grateful for all the wonderful things in life.",
        "Successfully completed my project, so proud!",
        "Dancing and singing with joy today!",
        "Life is amazing and full of possibilities.",
        "Celebrated with cake and laughter tonight.",

        # Sad (1)
        "Feeling really down and disappointed today.",
        "Missing my loved ones who are far away.",
        "Lost something important to me yesterday.",
        "Going through a difficult time right now.",
        "Everything feels overwhelming and hopeless.",
        "Tears are falling, heart feels heavy.",
        "Struggling to find motivation these days.",
        "Lonely and isolated from everyone.",
        "Remembering better times makes me melancholy.",
        "Wish things could be different somehow.",

        # Angry (2)
        "This is absolutely infuriating and unacceptable!",
        "Cannot believe how rude that person was!",
        "Traffic is driving me completely insane today.",
        "Fed up with all these constant problems.",
        "Extremely frustrated with this situation.",
        "How dare they treat me this way!",
        "Sick and tired of dealing with incompetence.",
        "This makes my blood boil with rage.",
        "Absolutely outraged by this injustice.",
        "Want to scream at the top of my lungs!",

        # Neutral (3)
        "The weather forecast shows rain tomorrow.",
        "Need to buy groceries after work today.",
        "Meeting scheduled for three o'clock.",
        "Reading a book about quantum physics.",
        "Coffee shop opens at seven in the morning.",
        "Taking the bus to downtown later.",
        "Researching different vacation destinations.",
        "Organizing files in my computer folder.",
        "Checking email and responding to messages.",
        "Planning next week's schedule and appointments."
    ]

    labels = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10

    return texts, labels, emotion_names


def split_dataset(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split dataset into train/validation/test sets.

    Args:
        features: Feature tensor
        labels: Label tensor
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed

    Returns:
        Dictionary with train/val/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Convert to numpy for sklearn
    X = features.numpy()
    y = labels.numpy()

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=y
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        random_state=random_state,
        stratify=y_temp
    )

    # Convert back to tensors
    return {
        'train': (torch.tensor(X_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.long)),
        'val': (torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)),
        'test': (torch.tensor(X_test, dtype=torch.float32),
                 torch.tensor(y_test, dtype=torch.long))
    }


def create_data_loaders(
    dataset_splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 32,
    shuffle_train: bool = True
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create PyTorch data loaders from dataset splits.

    Args:
        dataset_splits: Dictionary with train/val/test data
        batch_size: Batch size for data loaders
        shuffle_train: Whether to shuffle training data

    Returns:
        Dictionary with data loaders
    """
    loaders = {}

    for split_name, (features, labels) in dataset_splits.items():
        dataset = torch.utils.data.TensorDataset(features, labels)
        shuffle = shuffle_train if split_name == 'train' else False

        loaders[split_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )

    return loaders


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot comprehensive training history.

    Args:
        history: Training history dictionary
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy plot
    if 'train_accuracy' in history:
        axes[0, 1].plot(history['train_accuracy'], label='Train Acc', color='blue')
    if 'val_accuracy' in history:
        axes[0, 1].plot(history['val_accuracy'], label='Val Acc', color='red')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Quantum gradient norms
    if 'quantum_grad_norms' in history and history['quantum_grad_norms']:
        axes[1, 0].plot(history['quantum_grad_norms'], color='green', alpha=0.7)
        axes[1, 0].set_title('Quantum Gradient Norms')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

    # Classical gradient norms
    if 'classical_grad_norms' in history and history['classical_grad_norms']:
        axes[1, 1].plot(history['classical_grad_norms'], color='orange', alpha=0.7)
        axes[1, 1].set_title('Classical Gradient Norms')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_dataset_distribution(
    features: torch.Tensor,
    labels: torch.Tensor,
    emotion_names: List[str]
) -> None:
    """
    Plot dataset distribution and feature analysis.

    Args:
        features: Feature tensor
        labels: Label tensor
        emotion_names: List of emotion names
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Class distribution
    unique, counts = torch.unique(labels, return_counts=True)
    axes[0, 0].bar([emotion_names[i] for i in unique], counts)
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_ylabel('Count')

    # Feature correlation matrix
    corr_matrix = torch.corrcoef(features.T)
    im = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 1].set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=axes[0, 1])

    # Feature distributions by class
    for i, emotion in enumerate(emotion_names):
        class_features = features[labels == i]
        if len(class_features) > 0:
            axes[1, 0].hist(
                class_features[:, 0].numpy(),
                alpha=0.7,
                label=emotion,
                bins=20
            )
    axes[1, 0].set_title('Feature 0 Distribution by Class')
    axes[1, 0].set_xlabel('Feature Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # 2D feature scatter
    colors = ['red', 'blue', 'green', 'orange']
    for i, emotion in enumerate(emotion_names):
        class_features = features[labels == i]
        if len(class_features) > 0:
            axes[1, 1].scatter(
                class_features[:, 0],
                class_features[:, 1],
                alpha=0.7,
                label=emotion,
                color=colors[i % len(colors)]
            )
    axes[1, 1].set_title('Feature Space Visualization')
    axes[1, 1].set_xlabel('Feature 0')
    axes[1, 1].set_ylabel('Feature 1')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        predictions: Predicted labels
        targets: True labels

    Returns:
        Dictionary with various metrics
    """
    return {
        'accuracy': accuracy_score(targets, predictions),
        'f1_macro': f1_score(targets, predictions, average='macro'),
        'f1_micro': f1_score(targets, predictions, average='micro'),
        'precision_macro': precision_score(targets, predictions, average='macro'),
        'recall_macro': recall_score(targets, predictions, average='macro')
    }


def visualize_quantum_circuit(n_qubits: int = 4, n_layers: int = 2) -> None:
    """
    Visualize the quantum circuit architecture.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, params):
        # Data encoding
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(inputs[i] * np.pi, wires=i)

        # Variational layers
        for layer in range(n_layers):
            # Entangling
            for i in range(0, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

            # Rotations
            for i in range(n_qubits):
                qml.RX(params[layer, i, 0], wires=i)
                qml.RY(params[layer, i, 1], wires=i)
                qml.RZ(params[layer, i, 2], wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Create sample inputs
    inputs = np.random.rand(n_qubits)
    params = np.random.rand(n_layers, n_qubits, 3) * 0.1

    print("Quantum Circuit Architecture:")
    print("=" * 50)

    # Draw circuit
    try:
        fig, ax = qml.draw_mpl(circuit)(inputs, params)
        plt.show()
    except:
        print("Circuit visualization requires matplotlib backend")

    # Text representation
    print("\nText representation:")
    print(qml.draw(circuit)(inputs, params))


def quantum_fidelity(state1: torch.Tensor, state2: torch.Tensor) -> float:
    """
    Calculate quantum state fidelity.

    Args:
        state1: First quantum state
        state2: Second quantum state

    Returns:
        Fidelity value between 0 and 1
    """
    # Ensure states are normalized
    state1 = state1 / torch.norm(state1)
    state2 = state2 / torch.norm(state2)

    # Calculate fidelity
    overlap = torch.abs(torch.vdot(state1, state2))
    return overlap.item() ** 2


def save_model_results(
    model_info: Dict[str, Any],
    training_history: Dict[str, List[float]],
    evaluation_results: Dict[str, Any],
    save_path: str
) -> None:
    """
    Save comprehensive model results to file.

    Args:
        model_info: Model configuration and statistics
        training_history: Training history
        evaluation_results: Evaluation results
        save_path: Path to save results
    """
    results = {
        'model_info': model_info,
        'training_history': training_history,
        'evaluation_results': evaluation_results
    }

    # Convert tensors to lists for JSON serialization
    for key, value in results.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    results[key][k] = v.tolist()
                elif isinstance(v, np.ndarray):
                    results[key][k] = v.tolist()

    # Save as JSON
    import json
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {save_path}")
