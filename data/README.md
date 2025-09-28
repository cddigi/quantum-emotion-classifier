# Emotion Classification Datasets

This directory contains training and testing datasets for the quantum emotion classifier.

## Dataset Description

### Emotion Classes

The datasets contain four emotion classes:

- **0 - Happy**: Expressions of joy, excitement, gratitude, and positive emotions
- **1 - Sad**: Expressions of sadness, disappointment, grief, and melancholy
- **2 - Angry**: Expressions of anger, frustration, rage, and irritation
- **3 - Neutral**: Factual statements, observations, and emotionally neutral content

### Files

#### `emotions_train.csv`
- **Purpose**: Training dataset for model development
- **Size**: 60 samples (15 per emotion class)
- **Format**: CSV with columns `text` and `emotion`
- **Usage**: Use for training quantum emotion classifiers

#### `emotions_test.csv`
- **Purpose**: Testing dataset for model evaluation
- **Size**: 20 samples (5 per emotion class)
- **Format**: CSV with columns `text` and `emotion`
- **Usage**: Use for final evaluation and testing

### Data Characteristics

- **Language**: English
- **Text Length**: Varies from short phrases to complete sentences
- **Emotional Intensity**: Range from subtle to explicit emotional expressions
- **Balance**: Equal distribution across emotion classes
- **Quality**: Hand-crafted examples with clear emotional indicators

### Usage Example

```python
import pandas as pd
from src.quantum_emotion.encoding import TextEncoder
from src.quantum_emotion.classifier import QuantumTextClassifier

# Load training data
train_data = pd.read_csv('data/emotions_train.csv')
texts = train_data['text'].tolist()
labels = train_data['emotion'].tolist()

# Encode text features
encoder = TextEncoder(encoding_type='statistical')
features = encoder.fit_transform(texts)

# Train quantum classifier
model = QuantumTextClassifier(n_qubits=6, n_classes=4)
# ... training code ...
```

### Dataset Extensions

For larger experiments, consider:

1. **Synthetic Data Generation**: Use `create_emotion_dataset()` from utils
2. **Public Datasets**: Integrate with emotion datasets like GoEmotions or EmoBank
3. **Data Augmentation**: Generate variations of existing samples
4. **Multi-language**: Extend to other languages for cross-lingual studies

### Quantum Encoding Considerations

These text samples are designed to work well with quantum encoding methods:

- **Statistical Features**: Text length, word count, emotional indicators
- **TF-IDF Encoding**: Suitable for quantum amplitude encoding
- **Angle Encoding**: Feature values normalized for rotation angles
- **Kernel Methods**: Designed for quantum kernel similarity computation

### Citation

If using this dataset in research, please cite:

```
Quantum Emotion Classifier Dataset
Generated for quantum machine learning research
Available at: https://github.com/your-repo/quantum-emotion-classifier
```