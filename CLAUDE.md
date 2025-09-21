# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **quantum emotion classifier** project that will use PennyLane for quantum machine learning to classify text emotions. The project is currently in initial setup phase.

**Key Technologies:**
- Python 3.14rc3 (bleeding edge)
- uv package manager (Rust-powered, 10-100x faster than pip)
- PennyLane for quantum ML
- PyTorch for hybrid quantum-classical models
- Apple M3 Pro optimizations with Metal Performance Shaders

## Development Environment Setup

```bash
# Install uv package manager (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

# Create virtual environment with Python 3.14
uv venv --python 3.14
source .venv/bin/activate

# Initialize project structure
uv init .
```

## Project Structure (To Be Created)

```
quantum-emotion-classifier/
├── pyproject.toml              # uv project configuration
├── uv.lock                     # Locked dependencies
├── .python-version             # Pin Python 3.14rc3
├── src/
│   └── quantum_emotion/
│       ├── __init__.py
│       ├── classifier.py       # Main QuantumTextClassifier class
│       ├── kernels.py          # Quantum kernel methods
│       ├── hybrid.py           # PyTorch integration
│       ├── encoding.py         # Text to quantum state encoding
│       └── utils.py            # Helper functions
├── tests/
│   ├── test_circuits.py        # Quantum circuit tests
│   ├── test_gradients.py       # Gradient computation tests
│   └── test_classification.py  # End-to-end tests
├── notebooks/
│   ├── 01_quantum_basics.ipynb # Interactive tutorial
│   ├── 02_emotion_demo.ipynb   # Live demonstration
│   └── 03_benchmarks.ipynb     # Performance analysis
├── data/
│   ├── emotions_train.csv      # Training dataset
│   └── emotions_test.csv       # Test dataset
└── experiments/
    ├── scaling_analysis.py      # Qubit scaling experiments
    └── optimization_comparison.py # Classical vs quantum
```

## Core Dependencies

Essential packages to install:

```bash
# Core quantum ML stack
uv add pennylane torch numpy scikit-learn matplotlib

# PyTorch with Metal Performance Shaders (Apple Silicon)
uv add torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Development tools
uv add --dev pytest pytest-cov black ruff mypy

# Optional: Hardware quantum backends
uv add --optional pennylane-qiskit pennylane-cirq

# Optional: Jupyter notebooks
uv add --optional jupyter ipywidgets plotly
```

## Key Commands

### Development Workflow
```bash
# Add dependencies
uv add package_name

# Run tests
uv run pytest tests/ -v

# Format code
uv run black src/ tests/
uv run ruff check src/ tests/

# Type checking
uv run mypy src/

# Start Jupyter
uv run jupyter lab notebooks/

# Run specific experiments
uv run python experiments/scaling_analysis.py
```

### Testing Commands
```bash
# Run all tests with coverage
uv run pytest --cov=quantum_emotion tests/

# Run specific test file
uv run pytest tests/test_circuits.py -v

# Run benchmarks
uv run pytest tests/ --benchmark-only

# Run single test
uv run pytest tests/test_circuits.py::test_circuit_depth -v
```

## Architecture Overview

The quantum emotion classifier uses a hybrid quantum-classical approach:

1. **Classical Preprocessing**: Convert text → numerical features
2. **Quantum Encoding**: Map features → quantum state amplitudes
3. **Variational Quantum Circuit**: Learnable quantum transformations
4. **Measurement**: Extract quantum expectation values
5. **Classical Postprocessing**: Map to emotion probabilities

### Key Classes to Implement

```python
class QuantumTextClassifier:
    """Main classifier combining quantum and classical processing"""
    def __init__(self, n_qubits: int, n_classes: int, n_layers: int)
    def encode_features(self, text_features) -> QuantumState
    def forward(self, x) -> EmotionProbabilities
    def train(self, data, epochs: int)
    def predict(self, text: str) -> int

class QuantumKernel:
    """Quantum kernel methods for similarity computation"""
    def compute_kernel(self, x1, x2) -> float
    def kernel_matrix(self, X) -> Matrix

class HybridModel(torch.nn.Module):
    """PyTorch wrapper for quantum circuits with automatic differentiation"""
```

## Quantum Advantage Strategy

**Exponential Feature Compression**: Process text in 2^n dimensional Hilbert space using only n qubits
- Example: 20 qubits → 2^20 = 1,048,576 dimensional space
- Classical equivalent would need 1M+ parameters
- Quantum uses ~80 parameters (20 qubits × 4 rotation angles)

## Performance Targets

| Configuration | Qubits | Hilbert Dimension | Parameters | Target Accuracy |
|--------------|--------|-------------------|------------|-----------------|
| Desktop Demo | 10 | 1,024 | ~40 | 85%+ |
| Full Scale | 20 | 1,048,576 | ~80 | 90%+ |
| Maximum | 28 | 268M+ | ~112 | 92%+ |

## Python 3.14 Features to Leverage

- **Pattern Matching**: Use for circuit type selection
- **Type Expressions**: Enhanced quantum type hints
- **Performance**: ~15% faster execution, better memory management
- **F-string debugging**: `print(f"{learning_rate=:.4f}")`

## Apple M3 Pro Optimizations

```python
# Use Metal Performance Shaders for classical layers
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Optimize for M3 Pro performance cores
import os
os.environ['OMP_NUM_THREADS'] = '10'  # M3 Pro has 10 performance cores
```

## Testing Strategy

- **Unit Tests**: Individual quantum circuits and encodings
- **Integration Tests**: End-to-end classification pipeline
- **Gradient Tests**: Verify quantum gradient computation
- **Performance Tests**: Benchmark scaling with qubit count
- **Hardware Tests**: Validate on quantum simulators

## Data Requirements

**Training Data Format**:
```csv
text,emotion
"I am so happy today!",0  # Happy
"This makes me sad",1     # Sad
"I'm feeling angry",2     # Angry
"Just neutral feeling",3  # Neutral
```

**Features**: Convert text to numerical vectors (TF-IDF, embeddings, etc.)
**Labels**: Integer emotion categories (0-3 for 4 emotions)

## Development Guidelines

1. **Start Simple**: Begin with 4 qubits and 4 emotions
2. **Test First**: Write quantum circuit tests before implementation
3. **Gradual Scaling**: Increase qubit count once basic version works
4. **Hybrid Approach**: Use quantum where it provides clear advantage
5. **Profile Early**: Monitor memory usage and circuit depth

## Common Development Tasks

```bash
# Set up new environment
uv venv --python 3.14 && source .venv/bin/activate

# Quick prototype test
uv run python -c "import pennylane as qml; print('PennyLane ready!')"

# Add new feature
uv add package_name
uv run pytest tests/

# Performance check
uv run python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Experiment workflow
uv run python experiments/new_experiment.py
uv run pytest tests/test_new_feature.py -v
```

## Troubleshooting

**Python 3.14 not found**: `uv venv --python 3.14` (uv will download it)
**MPS not available**: Ensure running on Apple Silicon with macOS 12.3+
**Memory issues**: Reduce qubit count or use CPU-only mode
**Slow gradients**: Decrease circuit depth or batch size

This quantum emotion classifier demonstrates genuine quantum advantages while remaining practical for desktop development and Hacktoberfest contributions.