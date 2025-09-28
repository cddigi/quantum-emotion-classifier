"""
Text to quantum state encoding methods.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re


class TextEncoder:
    """
    Convert text into numerical features suitable for quantum encoding.

    Provides multiple encoding strategies from simple statistical features
    to TF-IDF vectors that can be embedded into quantum states.
    """

    def __init__(
        self,
        encoding_type: str = "statistical",
        max_features: int = 128,
        normalize: bool = True
    ):
        """
        Initialize text encoder.

        Args:
            encoding_type: Type of encoding ('statistical', 'tfidf', 'count')
            max_features: Maximum number of features
            normalize: Whether to normalize features
        """
        self.encoding_type = encoding_type
        self.max_features = max_features
        self.normalize = normalize

        # Initialize encoders
        if encoding_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
        elif encoding_type == "count":
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
        else:
            self.vectorizer = None

        self.scaler = StandardScaler() if normalize else None
        self.is_fitted = False

    def _extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """Extract statistical features from text."""
        features = []

        for text in texts:
            # Clean text
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            words = clean_text.split()

            # Basic statistics
            text_length = len(text)
            word_count = len(words)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            sentence_count = len(re.split(r'[.!?]+', text))

            # Character frequencies
            char_freq = {}
            for char in text.lower():
                if char.isalpha():
                    char_freq[char] = char_freq.get(char, 0) + 1

            total_chars = sum(char_freq.values())
            vowel_ratio = sum(char_freq.get(v, 0) for v in 'aeiou') / (total_chars + 1e-8)

            # Emotional indicators (simple keyword matching)
            positive_words = ['happy', 'joy', 'love', 'good', 'great', 'amazing', 'wonderful']
            negative_words = ['sad', 'angry', 'hate', 'bad', 'terrible', 'awful', 'horrible']

            positive_score = sum(1 for word in words if word in positive_words)
            negative_score = sum(1 for word in words if word in negative_words)

            # Punctuation analysis
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1e-8)

            feature_vector = [
                text_length / 100.0,  # Normalize
                word_count / 50.0,
                avg_word_length / 10.0,
                sentence_count / 10.0,
                vowel_ratio,
                positive_score / (word_count + 1e-8),
                negative_score / (word_count + 1e-8),
                exclamation_count / 10.0,
                question_count / 10.0,
                caps_ratio
            ]

            features.append(feature_vector)

        return np.array(features)

    def fit(self, texts: List[str]) -> 'TextEncoder':
        """
        Fit encoder to training texts.

        Args:
            texts: List of training texts

        Returns:
            Self for method chaining
        """
        if self.encoding_type == "statistical":
            # Extract statistical features for normalization
            features = self._extract_statistical_features(texts)
            if self.scaler:
                self.scaler.fit(features)
        else:
            # Fit vectorizer
            self.vectorizer.fit(texts)
            if self.scaler:
                features = self.vectorizer.transform(texts).toarray()
                self.scaler.fit(features)

        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to numerical features.

        Args:
            texts: List of texts to transform

        Returns:
            Numerical feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transform")

        if self.encoding_type == "statistical":
            features = self._extract_statistical_features(texts)
        else:
            features = self.vectorizer.transform(texts).toarray()

        # Apply normalization
        if self.scaler:
            features = self.scaler.transform(features)

        return features

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit encoder and transform texts in one step."""
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        if self.encoding_type == "statistical":
            return [
                'text_length', 'word_count', 'avg_word_length', 'sentence_count',
                'vowel_ratio', 'positive_score', 'negative_score',
                'exclamation_count', 'question_count', 'caps_ratio'
            ]
        elif hasattr(self.vectorizer, 'get_feature_names_out'):
            return list(self.vectorizer.get_feature_names_out())
        else:
            return [f'feature_{i}' for i in range(self.max_features)]


class QuantumStateEncoder:
    """
    Encode classical features into quantum states.

    Provides different quantum encoding strategies for mapping classical
    data to quantum states suitable for quantum machine learning.
    """

    def __init__(self, n_qubits: int, encoding_method: str = "angle"):
        """
        Initialize quantum state encoder.

        Args:
            n_qubits: Number of qubits available
            encoding_method: Encoding method ('angle', 'amplitude', 'basis')
        """
        self.n_qubits = n_qubits
        self.encoding_method = encoding_method
        self.max_dimension = 2 ** n_qubits

    def angle_encoding(self, features: np.ndarray) -> np.ndarray:
        """
        Angle encoding: map features to rotation angles.

        Args:
            features: Classical features

        Returns:
            Encoded features for angle encoding
        """
        # Ensure features fit in available qubits
        if len(features) > self.n_qubits:
            # Use PCA or truncation for dimensionality reduction
            features = features[:self.n_qubits]
        elif len(features) < self.n_qubits:
            # Pad with zeros
            padded = np.zeros(self.n_qubits)
            padded[:len(features)] = features
            features = padded

        # Scale to [0, 2π] for rotation angles
        features_scaled = np.clip(features, -1, 1)  # Assume normalized input
        return features_scaled

    def amplitude_encoding(self, features: np.ndarray) -> np.ndarray:
        """
        Amplitude encoding: map features to quantum state amplitudes.

        Args:
            features: Classical features

        Returns:
            Normalized amplitudes for quantum state
        """
        # Pad or truncate to fit 2^n_qubits dimension
        if len(features) > self.max_dimension:
            features = features[:self.max_dimension]
        elif len(features) < self.max_dimension:
            padded = np.zeros(self.max_dimension)
            padded[:len(features)] = features
            features = padded

        # Normalize to unit vector for valid quantum state
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        else:
            # Default to uniform superposition
            features = np.ones(self.max_dimension) / np.sqrt(self.max_dimension)

        return features

    def basis_encoding(self, features: np.ndarray) -> np.ndarray:
        """
        Basis encoding: map features to computational basis states.

        Args:
            features: Classical features (should be binary or discrete)

        Returns:
            Basis state indices
        """
        # Convert continuous features to discrete
        binary_features = (features > 0).astype(int)

        # Truncate to available qubits
        if len(binary_features) > self.n_qubits:
            binary_features = binary_features[:self.n_qubits]

        # Convert binary array to integer (basis state index)
        state_index = 0
        for i, bit in enumerate(binary_features):
            state_index += bit * (2 ** i)

        return state_index

    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode features using the specified method.

        Args:
            features: Classical features to encode

        Returns:
            Quantum-encoded features
        """
        if self.encoding_method == "angle":
            return self.angle_encoding(features)
        elif self.encoding_method == "amplitude":
            return self.amplitude_encoding(features)
        elif self.encoding_method == "basis":
            return self.basis_encoding(features)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")


def create_emotion_features(texts: List[str], emotions: List[str]) -> Dict[str, Any]:
    """
    Create comprehensive feature set for emotion classification.

    Args:
        texts: List of text samples
        emotions: List of emotion labels

    Returns:
        Dictionary with features and metadata
    """
    # Create multiple encoders
    encoders = {
        'statistical': TextEncoder('statistical'),
        'tfidf_small': TextEncoder('tfidf', max_features=32),
        'tfidf_large': TextEncoder('tfidf', max_features=128)
    }

    features = {}
    for name, encoder in encoders.items():
        features[name] = encoder.fit_transform(texts)

    # Emotion distribution analysis
    unique_emotions = list(set(emotions))
    emotion_counts = {emotion: emotions.count(emotion) for emotion in unique_emotions}

    # Text length distribution
    text_lengths = [len(text) for text in texts]

    return {
        'features': features,
        'encoders': encoders,
        'emotion_counts': emotion_counts,
        'text_lengths': text_lengths,
        'unique_emotions': unique_emotions,
        'feature_dimensions': {name: feat.shape[1] for name, feat in features.items()}
    }