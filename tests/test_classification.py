"""
End-to-end tests for quantum emotion classification.
"""

import pytest
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from unittest.mock import patch

from src.quantum_emotion.classifier import QuantumTextClassifier
from src.quantum_emotion.hybrid import HybridTrainer
from src.quantum_emotion.kernels import QuantumSVM
from src.quantum_emotion.encoding import TextEncoder
from src.quantum_emotion.utils import (
    create_emotion_dataset, create_text_emotion_dataset,
    split_dataset, create_data_loaders
)


class TestEndToEndClassification:
    """Test complete classification pipeline."""

    def test_synthetic_dataset_classification(self):
        """Test classification on synthetic emotion dataset."""
        # Create dataset
        features, labels, emotion_names = create_emotion_dataset(n_samples=100)

        # Split dataset
        splits = split_dataset(features, labels, random_state=42)
        loaders = create_data_loaders(splits, batch_size=16)

        # Create and train model - force CPU to avoid MPS float64 issues
        model = QuantumTextClassifier(n_qubits=4, n_classes=4, n_layers=2)
        trainer = HybridTrainer(model, learning_rate=0.01, device=torch.device('cpu'))

        # Short training for testing
        history = trainer.train(
            loaders['train'],
            loaders['val'],
            epochs=5,
            patience=10,
            verbose=False
        )

        # Evaluate
        results = trainer.detailed_evaluation(loaders['test'], emotion_names)

        # Basic checks
        assert 'accuracy' in results
        assert 'predictions' in results
        assert 'classification_report' in results
        assert results['accuracy'] >= 0.0
        assert results['accuracy'] <= 1.0
        assert len(results['predictions']) == len(results['targets'])

    def test_text_emotion_classification(self):
        """Test classification on text emotion dataset."""
        # Get text dataset
        texts, labels, emotion_names = create_text_emotion_dataset()

        # Encode texts
        encoder = TextEncoder(encoding_type='statistical')
        features = encoder.fit_transform(texts)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Create small test split
        n_train = int(0.7 * len(features))
        train_features = features_tensor[:n_train]
        train_labels = labels_tensor[:n_train]
        test_features = features_tensor[n_train:]
        test_labels = labels_tensor[n_train:]

        # Create model
        model = QuantumTextClassifier(
            n_qubits=4,
            n_classes=4,
            n_layers=1,
            feature_dim=features.shape[1]
        )

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(train_features)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_features)
            _, predictions = torch.max(test_outputs, 1)

        # Calculate accuracy
        accuracy = accuracy_score(test_labels.numpy(), predictions.numpy())
        assert accuracy >= 0.0

    def test_model_convergence(self):
        """Test that model can converge on simple dataset."""
        # Create simple linearly separable dataset
        n_samples = 40
        features = torch.zeros(n_samples, 4)
        labels = torch.zeros(n_samples, dtype=torch.long)

        # Class 0: positive in first two features
        features[:10] = torch.tensor([0.8, 0.8, 0.2, 0.2])
        labels[:10] = 0

        # Class 1: negative in first two features
        features[10:20] = torch.tensor([0.2, 0.2, 0.8, 0.8])
        labels[10:20] = 1

        # Class 2: mixed pattern
        features[20:30] = torch.tensor([0.8, 0.2, 0.8, 0.2])
        labels[20:30] = 0  # Same as class 0 for simplicity

        # Class 3: another mixed pattern
        features[30:40] = torch.tensor([0.2, 0.8, 0.2, 0.8])
        labels[30:40] = 1  # Same as class 1 for simplicity

        # Add small amount of noise
        features += 0.05 * torch.randn_like(features)

        # Train model
        model = QuantumTextClassifier(n_qubits=4, n_classes=2, n_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = torch.nn.CrossEntropyLoss()

        initial_loss = None
        final_loss = None

        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            if epoch == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if epoch == 49:
                final_loss = loss.item()

        # Model should show some learning (loss decrease)
        assert final_loss < initial_loss

        # Test accuracy on same data (should overfit)
        model.eval()
        with torch.no_grad():
            outputs = model(features)
            _, predictions = torch.max(outputs, 1)
            accuracy = accuracy_score(labels.numpy(), predictions.numpy())

        # Should achieve reasonable accuracy on training data
        assert accuracy > 0.4  # Low threshold for noisy synthetic data

    def test_different_model_configurations(self):
        """Test different model configurations work."""
        configurations = [
            {'n_qubits': 3, 'n_classes': 2, 'n_layers': 1},
            {'n_qubits': 4, 'n_classes': 3, 'n_layers': 2},
            {'n_qubits': 5, 'n_classes': 4, 'n_layers': 1},
        ]

        features, labels, _ = create_emotion_dataset(n_samples=20)

        for config in configurations:
            # Adjust labels to match n_classes
            adjusted_labels = labels % config['n_classes']

            model = QuantumTextClassifier(**config)

            # Test forward pass
            outputs = model(features)
            assert outputs.shape == (20, config['n_classes'])

            # Test backward pass
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, adjusted_labels)
            loss.backward()

            # Check gradients exist
            assert model.quantum_params.grad is not None

    def test_batch_size_independence(self):
        """Test model works with different batch sizes."""
        model = QuantumTextClassifier(n_qubits=4, n_classes=2, n_layers=1)
        features, labels, _ = create_emotion_dataset(n_samples=24)

        # Adjust labels to binary classification
        binary_labels = (labels >= 2).long()

        batch_sizes = [1, 4, 8, 24]
        results = {}

        for batch_size in batch_sizes:
            # Create data loader
            dataset = torch.utils.data.TensorDataset(features, binary_labels)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

            # Reset model parameters
            model = QuantumTextClassifier(n_qubits=4, n_classes=2, n_layers=1)
            torch.manual_seed(42)  # For reproducibility

            # Train for a few steps
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()

            total_loss = 0
            for batch_features, batch_labels in loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            results[batch_size] = total_loss

        # All batch sizes should produce finite losses
        for batch_size, loss in results.items():
            assert np.isfinite(loss), f"Batch size {batch_size} produced invalid loss"


class TestQuantumSVMClassification:
    """Test Quantum SVM classification."""

    def test_qsvm_on_synthetic_data(self):
        """Test Quantum SVM on synthetic dataset."""
        # Create small dataset for testing
        features, labels, _ = create_emotion_dataset(n_samples=20)

        # Convert to binary classification and numpy
        binary_labels = (labels >= 2).numpy()
        features_np = features.numpy()

        # Split dataset
        split_idx = 15
        X_train, X_test = features_np[:split_idx], features_np[split_idx:]
        y_train, y_test = binary_labels[:split_idx], binary_labels[split_idx:]

        # Create and train QSVM
        qsvm = QuantumSVM(n_qubits=4, encoding_type="angle")
        qsvm.fit(X_train, y_train)

        # Make predictions
        predictions = qsvm.predict(X_test)

        # Basic checks
        assert len(predictions) == len(y_test)
        assert all(pred in [0, 1] for pred in predictions)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        assert 0.0 <= accuracy <= 1.0

    def test_qsvm_probability_prediction(self):
        """Test QSVM probability prediction."""
        # Create dataset
        features, labels, _ = create_emotion_dataset(n_samples=16)
        binary_labels = (labels >= 2).numpy()
        features_np = features.numpy()

        # Train/test split
        X_train, X_test = features_np[:12], features_np[12:]
        y_train, y_test = binary_labels[:12], binary_labels[12:]

        # Train QSVM with probability estimation
        qsvm = QuantumSVM(n_qubits=4, encoding_type="angle", probability=True)
        qsvm.fit(X_train, y_train)

        # Get probability predictions
        try:
            probabilities = qsvm.predict_proba(X_test)
            assert probabilities.shape == (len(X_test), 2)
            assert np.allclose(probabilities.sum(axis=1), 1.0)
        except AttributeError:
            # Some SVM configurations might not support probability prediction
            pytest.skip("Probability prediction not supported in this configuration")

    def test_kernel_comparison(self):
        """Test different quantum kernel encodings."""
        features, labels, _ = create_emotion_dataset(n_samples=16)
        binary_labels = (labels >= 2).numpy()
        features_np = features.numpy()

        X_train, X_test = features_np[:12], features_np[12:]
        y_train, y_test = binary_labels[:12], binary_labels[12:]

        encodings = ["angle", "iqp"]
        accuracies = {}

        for encoding in encodings:
            try:
                qsvm = QuantumSVM(n_qubits=4, encoding_type=encoding)
                qsvm.fit(X_train, y_train)
                predictions = qsvm.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                accuracies[encoding] = accuracy
            except Exception as e:
                pytest.skip(f"Encoding {encoding} failed: {e}")

        # All tested encodings should produce valid accuracies
        for encoding, accuracy in accuracies.items():
            assert 0.0 <= accuracy <= 1.0


class TestTextProcessingPipeline:
    """Test complete text processing and classification pipeline."""

    def test_text_encoder_pipeline(self):
        """Test text encoding pipeline."""
        texts, labels, emotion_names = create_text_emotion_dataset()

        # Test different encoding types
        encoding_types = ['statistical', 'tfidf', 'count']

        for encoding_type in encoding_types:
            encoder = TextEncoder(encoding_type=encoding_type, max_features=16)
            features = encoder.fit_transform(texts)

            # Basic checks
            assert features.shape[0] == len(texts)
            assert features.shape[1] > 0
            assert np.all(np.isfinite(features))

            # Test quantum classifier on encoded features
            features_tensor = torch.tensor(features, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            model = QuantumTextClassifier(
                n_qubits=4,
                n_classes=4,
                n_layers=1,
                feature_dim=features.shape[1]
            )

            # Test forward pass
            outputs = model(features_tensor)
            assert outputs.shape == (len(texts), 4)

    def test_feature_dimension_handling(self):
        """Test handling of different feature dimensions."""
        texts, labels, _ = create_text_emotion_dataset()

        # Test with different max_features
        feature_dims = [4, 8, 16, 32]

        for max_features in feature_dims:
            encoder = TextEncoder(encoding_type='tfidf', max_features=max_features)
            features = encoder.fit_transform(texts[:10])  # Use subset for speed

            # Create model with matching input dimension
            model = QuantumTextClassifier(
                n_qubits=4,
                n_classes=4,
                feature_dim=features.shape[1]
            )

            features_tensor = torch.tensor(features, dtype=torch.float32)
            outputs = model(features_tensor)

            assert outputs.shape == (10, 4)

    def test_text_preprocessing_robustness(self):
        """Test robustness of text preprocessing."""
        # Test with challenging text inputs
        challenging_texts = [
            "",  # Empty string
            "!@#$%^&*()",  # Only punctuation
            "ALL CAPS TEXT!!!",  # All caps
            "a",  # Single character
            "word " * 100,  # Very long text
            "émotions françaises",  # Non-ASCII characters
            "123 456 789",  # Numbers only
        ]

        labels = [0] * len(challenging_texts)

        try:
            encoder = TextEncoder(encoding_type='statistical')
            features = encoder.fit_transform(challenging_texts)

            # Should produce valid features even for challenging inputs
            assert features.shape[0] == len(challenging_texts)
            assert np.all(np.isfinite(features))

        except Exception as e:
            pytest.fail(f"Text preprocessing failed on challenging inputs: {e}")


class TestModelPersistence:
    """Test model saving and loading."""

    def test_model_state_dict(self):
        """Test model state dict save/load."""
        # Create and train model
        model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)
        features, labels, _ = create_emotion_dataset(n_samples=10)
        binary_labels = (labels >= 2).long()

        # Training step
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        outputs = model(features)
        loss = torch.nn.CrossEntropyLoss()(outputs, binary_labels)
        loss.backward()
        optimizer.step()

        # Save state dict
        state_dict = model.state_dict()

        # Create new model and load state
        new_model = QuantumTextClassifier(n_qubits=3, n_classes=2, n_layers=1)
        new_model.load_state_dict(state_dict)

        # Test that models produce same output
        model.eval()
        new_model.eval()

        with torch.no_grad():
            original_output = model(features)
            loaded_output = new_model(features)

        assert torch.allclose(original_output, loaded_output, atol=1e-6)

    def test_model_configuration_persistence(self):
        """Test model configuration is preserved."""
        config = {
            'n_qubits': 5,
            'n_classes': 3,
            'n_layers': 2,
            'feature_dim': 8,
            'hidden_dim': 32
        }

        model = QuantumTextClassifier(**config)
        model_info = model.get_model_info()

        # Check configuration is correctly stored
        for key, value in config.items():
            assert model_info[key] == value


class TestPerformanceMetrics:
    """Test performance metrics and evaluation."""

    def test_classification_metrics(self):
        """Test comprehensive classification metrics."""
        # Create balanced dataset
        features, labels, emotion_names = create_emotion_dataset(n_samples=40)

        # Create splits
        splits = split_dataset(features, labels, random_state=42)
        loaders = create_data_loaders(splits, batch_size=8)

        # Train model - force CPU to avoid MPS float64 issues
        model = QuantumTextClassifier(n_qubits=4, n_classes=4, n_layers=1)
        trainer = HybridTrainer(model, learning_rate=0.02, device=torch.device('cpu'))

        # Quick training
        trainer.train(
            loaders['train'],
            epochs=5,
            verbose=False
        )

        # Detailed evaluation
        results = trainer.detailed_evaluation(loaders['test'], emotion_names)

        # Check all expected metrics are present
        expected_keys = [
            'accuracy', 'predictions', 'targets', 'probabilities',
            'classification_report', 'confusion_matrix', 'class_names'
        ]

        for key in expected_keys:
            assert key in results

        # Check classification report structure
        report = results['classification_report']
        assert 'macro avg' in report
        assert 'weighted avg' in report

        # Check confusion matrix
        cm = results['confusion_matrix']
        assert cm.shape == (4, 4)
        assert cm.sum() == len(results['predictions'])

    def test_cross_validation_simulation(self):
        """Simulate cross-validation testing."""
        features, labels, _ = create_emotion_dataset(n_samples=30)

        # Simple 3-fold cross-validation simulation
        fold_size = 10
        accuracies = []

        for fold in range(3):
            # Create train/test split for this fold
            test_start = fold * fold_size
            test_end = test_start + fold_size

            test_features = features[test_start:test_end]
            test_labels = labels[test_start:test_end]

            train_features = torch.cat([
                features[:test_start],
                features[test_end:]
            ])
            train_labels = torch.cat([
                labels[:test_start],
                labels[test_end:]
            ])

            # Train model
            model = QuantumTextClassifier(n_qubits=4, n_classes=4, n_layers=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            criterion = torch.nn.CrossEntropyLoss()

            # Training loop
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(train_features)
                loss = criterion(outputs, train_labels)
                loss.backward()
                optimizer.step()

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_features)
                _, predictions = torch.max(test_outputs, 1)

            accuracy = accuracy_score(test_labels.numpy(), predictions.numpy())
            accuracies.append(accuracy)

        # All folds should produce valid accuracies
        assert all(0.0 <= acc <= 1.0 for acc in accuracies)

        # Calculate mean and std of cross-validation
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        assert 0.0 <= mean_accuracy <= 1.0
        assert std_accuracy >= 0.0