#!/usr/bin/env python3
"""
Generate additional emotion datasets for quantum machine learning experiments.
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Dict
import argparse
import json
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.quantum_emotion.utils import create_emotion_dataset, create_text_emotion_dataset
from src.quantum_emotion.encoding import TextEncoder


def generate_synthetic_numerical_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    noise_level: float = 0.1,
    output_path: str = "synthetic_emotions.csv"
) -> None:
    """
    Generate synthetic numerical emotion dataset.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        noise_level: Amount of noise to add
        output_path: Output file path
    """
    print(f"Generating synthetic dataset with {n_samples} samples, {n_features} features...")

    # Create emotion prototypes in feature space
    emotion_prototypes = {
        0: np.array([0.8, 0.9, 0.1, 0.2, 0.7, 0.8, 0.2, 0.1, 0.9, 0.8]),  # Happy
        1: np.array([0.2, 0.1, 0.8, 0.9, 0.3, 0.2, 0.8, 0.9, 0.1, 0.2]),  # Sad
        2: np.array([0.6, 0.4, 0.6, 0.4, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),  # Angry
        3: np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])   # Neutral
    }

    # Ensure we have enough features
    for emotion in emotion_prototypes:
        if len(emotion_prototypes[emotion]) < n_features:
            # Extend with random values
            current_len = len(emotion_prototypes[emotion])
            extension = np.random.rand(n_features - current_len)
            emotion_prototypes[emotion] = np.concatenate([
                emotion_prototypes[emotion], extension
            ])
        else:
            emotion_prototypes[emotion] = emotion_prototypes[emotion][:n_features]

    samples_per_class = n_samples // 4
    data = []

    for emotion_idx in range(4):
        prototype = emotion_prototypes[emotion_idx]

        for _ in range(samples_per_class):
            # Add Gaussian noise around prototype
            sample = prototype + np.random.normal(0, noise_level, n_features)
            sample = np.clip(sample, 0, 1)  # Keep in valid range

            # Create feature names
            feature_names = [f"feature_{i}" for i in range(n_features)]

            # Create row
            row = {name: value for name, value in zip(feature_names, sample)}
            row['emotion'] = emotion_idx
            data.append(row)

    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved synthetic dataset to {output_path}")

    # Print statistics
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {n_features}")
    print(f"  Class distribution:")
    for emotion in range(4):
        count = len(df[df['emotion'] == emotion])
        print(f"    Emotion {emotion}: {count} samples")


def generate_expanded_text_dataset(
    base_multiplier: int = 5,
    output_path: str = "emotions_expanded.csv"
) -> None:
    """
    Generate expanded text emotion dataset with variations.

    Args:
        base_multiplier: How many variations to create per base sample
        output_path: Output file path
    """
    print(f"Generating expanded text dataset with {base_multiplier}x multiplier...")

    # Base emotion templates
    emotion_templates = {
        0: [  # Happy
            "I feel {intensity} about {subject}!",
            "What a {adjective} {time_period}, feeling {emotion}!",
            "This {event} makes me feel {positive_emotion}.",
            "{achievement} fills me with {feeling}.",
            "Sharing {positive_thing} and spreading {emotion}."
        ],
        1: [  # Sad
            "Feeling {intensity} and {emotion} about {subject}.",
            "Missing {subject} who {situation}.",
            "Going through {difficult_thing} right now.",
            "This {negative_event} makes me feel {sad_emotion}.",
            "Everything feels {negative_adjective} and {hopeless_word}."
        ],
        2: [  # Angry
            "This is absolutely {intensity} and {unacceptable_word}!",
            "Cannot believe how {negative_adjective} {subject} was!",
            "{situation} is driving me {intensity}!",
            "Fed up with all these {problems}.",
            "This makes me feel {angry_emotion} and {frustrated_word}."
        ],
        3: [  # Neutral
            "The {subject} {neutral_action} {time_reference}.",
            "Need to {task} {time_reference}.",
            "{activity} is scheduled for {time}.",
            "Currently {action} {object}.",
            "{routine_activity} {time_reference}."
        ]
    }

    # Word banks for templates
    word_banks = {
        'intensity': ['extremely', 'really', 'very', 'incredibly', 'absolutely'],
        'adjective': ['wonderful', 'beautiful', 'amazing', 'fantastic', 'perfect'],
        'time_period': ['day', 'morning', 'evening', 'moment', 'experience'],
        'emotion': ['happy', 'joyful', 'excited', 'grateful', 'blessed'],
        'event': ['news', 'opportunity', 'achievement', 'surprise', 'moment'],
        'positive_emotion': ['thrilled', 'overjoyed', 'elated', 'delighted', 'euphoric'],
        'achievement': ['Success', 'Victory', 'Accomplishment', 'Progress', 'Breakthrough'],
        'feeling': ['pride', 'joy', 'satisfaction', 'happiness', 'excitement'],
        'positive_thing': ['good vibes', 'positivity', 'happiness', 'joy', 'love'],
        'subject': ['everything', 'this situation', 'recent events', 'what happened', 'this'],
        'situation': ['are far away', 'left', 'moved away', 'are gone', 'departed'],
        'difficult_thing': ['a tough time', 'challenges', 'hardship', 'struggles', 'difficulties'],
        'negative_event': ['loss', 'disappointment', 'setback', 'failure', 'rejection'],
        'sad_emotion': ['heartbroken', 'devastated', 'crushed', 'defeated', 'hopeless'],
        'negative_adjective': ['terrible', 'awful', 'horrible', 'disappointing', 'devastating'],
        'hopeless_word': ['hopeless', 'impossible', 'overwhelming', 'unbearable', 'crushing'],
        'unacceptable_word': ['unacceptable', 'outrageous', 'ridiculous', 'infuriating', 'intolerable'],
        'problems': ['problems', 'issues', 'complications', 'obstacles', 'challenges'],
        'angry_emotion': ['furious', 'livid', 'enraged', 'irate', 'incensed'],
        'frustrated_word': ['frustrated', 'annoyed', 'irritated', 'aggravated', 'exasperated'],
        'neutral_action': ['shows', 'indicates', 'suggests', 'reports', 'states'],
        'time_reference': ['today', 'tomorrow', 'this week', 'later', 'soon'],
        'task': ['complete', 'finish', 'handle', 'organize', 'review'],
        'activity': ['Meeting', 'Conference', 'Appointment', 'Session', 'Review'],
        'time': ['3 PM', 'morning', 'afternoon', 'next week', 'Friday'],
        'action': ['reading', 'reviewing', 'organizing', 'checking', 'updating'],
        'object': ['documents', 'files', 'reports', 'emails', 'data'],
        'routine_activity': ['Planning', 'Scheduling', 'Organizing', 'Preparing', 'Reviewing']
    }

    generated_texts = []
    generated_labels = []

    for emotion in range(4):
        templates = emotion_templates[emotion]

        for _ in range(base_multiplier * 10):  # 10 base samples per emotion
            template = np.random.choice(templates)

            # Fill in template with random words
            filled_template = template
            for placeholder, words in word_banks.items():
                if f'{{{placeholder}}}' in filled_template:
                    word = np.random.choice(words)
                    filled_template = filled_template.replace(f'{{{placeholder}}}', word)

            generated_texts.append(filled_template)
            generated_labels.append(emotion)

    # Create DataFrame
    df = pd.DataFrame({
        'text': generated_texts,
        'emotion': generated_labels
    })

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved expanded text dataset to {output_path}")

    # Print statistics
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Class distribution:")
    emotion_names = ['Happy', 'Sad', 'Angry', 'Neutral']
    for emotion in range(4):
        count = len(df[df['emotion'] == emotion])
        print(f"    {emotion_names[emotion]}: {count} samples")


def create_feature_analysis_dataset(output_path: str = "feature_analysis.json") -> None:
    """
    Create dataset with extracted features for analysis.

    Args:
        output_path: Output JSON file path
    """
    print("Creating feature analysis dataset...")

    # Load existing text data
    train_df = pd.read_csv("emotions_train.csv")
    test_df = pd.read_csv("emotions_test.csv")

    all_texts = train_df['text'].tolist() + test_df['text'].tolist()
    all_labels = train_df['emotion'].tolist() + test_df['emotion'].tolist()

    # Extract features using different encoders
    encoders = {
        'statistical': TextEncoder('statistical'),
        'tfidf_16': TextEncoder('tfidf', max_features=16),
        'tfidf_32': TextEncoder('tfidf', max_features=32),
        'count_16': TextEncoder('count', max_features=16)
    }

    feature_analysis = {
        'texts': all_texts,
        'labels': all_labels,
        'features': {},
        'feature_names': {},
        'statistics': {}
    }

    for name, encoder in encoders.items():
        print(f"  Extracting {name} features...")
        features = encoder.fit_transform(all_texts)
        feature_names = encoder.get_feature_names()

        feature_analysis['features'][name] = features.tolist()
        feature_analysis['feature_names'][name] = feature_names
        feature_analysis['statistics'][name] = {
            'shape': features.shape,
            'mean': np.mean(features, axis=0).tolist(),
            'std': np.std(features, axis=0).tolist(),
            'min': np.min(features, axis=0).tolist(),
            'max': np.max(features, axis=0).tolist()
        }

    # Save analysis
    with open(output_path, 'w') as f:
        json.dump(feature_analysis, f, indent=2)

    print(f"Saved feature analysis to {output_path}")


def create_quantum_benchmark_dataset(
    n_qubits_list: List[int] = [3, 4, 5, 6],
    output_dir: str = "quantum_benchmarks"
) -> None:
    """
    Create datasets optimized for different quantum configurations.

    Args:
        n_qubits_list: List of qubit counts to optimize for
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Creating quantum benchmark datasets for {n_qubits_list} qubits...")

    for n_qubits in n_qubits_list:
        print(f"  Creating dataset for {n_qubits} qubits...")

        # Create dataset with appropriate number of features
        n_samples = min(1000, 2**n_qubits * 10)  # Scale with Hilbert space size
        features, labels, emotion_names = create_emotion_dataset(
            n_samples=n_samples,
            noise_level=0.1,
            random_state=42
        )

        # Adjust feature dimension to match quantum encoding capacity
        if features.shape[1] < n_qubits:
            # Pad with zeros
            padding = torch.zeros(features.shape[0], n_qubits - features.shape[1])
            features = torch.cat([features, padding], dim=1)
        elif features.shape[1] > n_qubits:
            # Truncate
            features = features[:, :n_qubits]

        # Save as CSV
        df = pd.DataFrame(features.numpy())
        df.columns = [f"feature_{i}" for i in range(n_qubits)]
        df['emotion'] = labels.numpy()

        output_file = output_path / f"quantum_{n_qubits}qubits.csv"
        df.to_csv(output_file, index=False)

        # Save metadata
        metadata = {
            'n_qubits': n_qubits,
            'n_samples': n_samples,
            'n_features': n_qubits,
            'n_classes': 4,
            'hilbert_dimension': 2**n_qubits,
            'compression_ratio': (2**n_qubits) / (n_qubits * 3),  # Assuming 3 params per qubit
            'emotion_names': emotion_names,
            'description': f'Quantum emotion dataset optimized for {n_qubits} qubits'
        }

        metadata_file = output_path / f"quantum_{n_qubits}qubits_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f"Saved quantum benchmark datasets to {output_dir}/")


def main():
    """Main function to generate all datasets."""
    parser = argparse.ArgumentParser(description="Generate emotion datasets for quantum ML")
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic numerical dataset')
    parser.add_argument('--expanded-text', action='store_true', help='Generate expanded text dataset')
    parser.add_argument('--feature-analysis', action='store_true', help='Create feature analysis dataset')
    parser.add_argument('--quantum-benchmarks', action='store_true', help='Create quantum benchmark datasets')
    parser.add_argument('--all', action='store_true', help='Generate all datasets')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples for synthetic dataset')
    parser.add_argument('--features', type=int, default=10, help='Number of features for synthetic dataset')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise level for synthetic dataset')
    parser.add_argument('--multiplier', type=int, default=5, help='Multiplier for expanded text dataset')

    args = parser.parse_args()

    if args.all or args.synthetic:
        generate_synthetic_numerical_dataset(
            n_samples=args.samples,
            n_features=args.features,
            noise_level=args.noise
        )

    if args.all or args.expanded_text:
        generate_expanded_text_dataset(base_multiplier=args.multiplier)

    if args.all or args.feature_analysis:
        create_feature_analysis_dataset()

    if args.all or args.quantum_benchmarks:
        create_quantum_benchmark_dataset()

    print("Dataset generation complete!")


if __name__ == "__main__":
    main()