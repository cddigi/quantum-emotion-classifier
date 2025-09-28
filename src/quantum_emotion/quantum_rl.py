"""
Quantum Reinforcement Learning for Atari Games

Implementation of various quantum RL algorithms that can be benchmarked
against classical deep RL approaches like QRDQN. Designed specifically
for Space Invaders comparison with existing classical baselines.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import gymnasium as gym
from collections import deque
import random
import warnings


class QuantumFeatureEncoder:
    """
    Quantum feature encoder for Atari game states.
    Compresses high-dimensional visual features into quantum states.
    """

    def __init__(self, n_qubits: int = 8, encoding_type: str = "amplitude"):
        """
        Initialize quantum feature encoder.

        Args:
            n_qubits: Number of qubits (determines feature space: 2^n dimensions)
            encoding_type: Type of encoding ('amplitude', 'angle', 'basis')
        """
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        self.feature_dim = 2 ** n_qubits  # Quantum feature space dimension

    def preprocess_atari_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess Atari frame for quantum encoding.

        Args:
            frame: Atari frame (84x84x4 typically after Atari wrapper)

        Returns:
            Compressed feature vector suitable for quantum encoding
        """
        # Flatten and downsample the frame
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)  # Convert to grayscale if needed

        # Resize to manageable feature vector
        from scipy.ndimage import zoom
        target_size = min(64, self.feature_dim)  # Limit to quantum feature space
        scale_factor = target_size / frame.size

        if scale_factor < 1:
            downsampled = zoom(frame.flatten(), scale_factor)
        else:
            downsampled = frame.flatten()[:target_size]

        # Normalize to [0, 1] for quantum encoding
        normalized = (downsampled - downsampled.min()) / (downsampled.max() - downsampled.min() + 1e-8)

        # Pad or truncate to exact quantum feature dimension
        if len(normalized) < self.feature_dim:
            padded = np.zeros(self.feature_dim)
            padded[:len(normalized)] = normalized
            normalized = padded
        else:
            normalized = normalized[:self.feature_dim]

        return normalized

    def encode_to_quantum_state(self, features: np.ndarray) -> np.ndarray:
        """
        Encode classical features into quantum state preparation parameters.

        Args:
            features: Classical feature vector

        Returns:
            Quantum encoding parameters
        """
        if self.encoding_type == "amplitude":
            # Amplitude encoding: normalize for quantum state
            normalized = features / (np.linalg.norm(features) + 1e-8)
            return normalized

        elif self.encoding_type == "angle":
            # Angle encoding: map features to rotation angles
            return features * np.pi

        elif self.encoding_type == "basis":
            # Basis encoding: binary representation
            binary_features = (features > 0.5).astype(float)
            return binary_features

        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")


class QuantumDQN(nn.Module):
    """
    Quantum Deep Q-Network combining quantum circuits with classical processing.
    Designed to be a drop-in replacement for classical DQN in RL environments.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_actions: int = 6,  # Space Invaders has 6 actions
        n_layers: int = 3,
        backend: str = "default.qubit",
        classical_hidden_dim: int = 128
    ):
        """
        Initialize Quantum DQN.

        Args:
            n_qubits: Number of qubits for quantum processing
            n_actions: Number of possible actions
            n_layers: Number of variational quantum layers
            backend: PennyLane quantum backend
            classical_hidden_dim: Hidden dimension for classical layers
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.classical_hidden_dim = classical_hidden_dim

        # Initialize quantum device
        self.device = qml.device(backend, wires=n_qubits)

        # Quantum parameters - learnable variational parameters
        self.quantum_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1  # 3 rotation gates per qubit
        )

        # Feature encoder
        self.encoder = QuantumFeatureEncoder(n_qubits, encoding_type="angle")

        # Classical preprocessing (like CNN feature extraction)
        self.classical_preprocess = nn.Sequential(
            nn.Linear(84 * 84, 512),  # Atari frame size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2 ** n_qubits),  # Map to quantum feature space
            nn.Tanh()  # Bound for angle encoding
        )

        # Classical postprocessing
        self.classical_postprocess = nn.Sequential(
            nn.Linear(n_qubits, classical_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(classical_hidden_dim, classical_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(classical_hidden_dim // 2, n_actions)
        )

        # Create quantum node
        self.qnode = qml.QNode(self._quantum_circuit, self.device, diff_method="parameter-shift")

    def _quantum_circuit(self, inputs: torch.Tensor, params: torch.Tensor) -> List[float]:
        """
        Variational quantum circuit for Q-value computation.

        Args:
            inputs: Quantum-encoded input features
            params: Variational parameters

        Returns:
            Expectation values from quantum measurements
        """
        # Data encoding layer
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            if i < len(inputs):
                qml.RY(inputs[i] * np.pi, wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            # Entangling layer
            for i in range(0, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

            # Parameterized layer
            for i in range(self.n_qubits):
                qml.RX(params[layer, i, 0], wires=i)
                qml.RY(params[layer, i, 1], wires=i)
                qml.RZ(params[layer, i, 2], wires=i)

        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum DQN.

        Args:
            x: Input state (batch_size, channels, height, width) or flattened

        Returns:
            Q-values for each action
        """
        batch_size = x.shape[0]

        # Flatten input if needed (handle Atari frames)
        if len(x.shape) > 2:
            x = x.view(batch_size, -1)

        # Classical preprocessing
        classical_features = self.classical_preprocess(x)

        # Process each sample in batch through quantum circuit
        quantum_outputs = []
        for i in range(batch_size):
            # Quantum processing
            q_out = self.qnode(classical_features[i], self.quantum_params)

            # Convert to tensor preserving gradients
            q_tensor = torch.stack([
                torch.as_tensor(val, dtype=torch.float32, device=x.device)
                for val in q_out
            ])
            quantum_outputs.append(q_tensor)

        # Stack batch results
        quantum_batch = torch.stack(quantum_outputs)

        # Classical postprocessing to get Q-values
        q_values = self.classical_postprocess(quantum_batch)

        return q_values


class HybridQuantumDQN(nn.Module):
    """
    Hybrid approach that combines classical CNN features with quantum processing.
    More practical for Atari games with complex visual inputs.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_actions: int = 6,
        n_layers: int = 2,
        backend: str = "lightning.qubit"
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers

        # Classical CNN for feature extraction (like in classical DQN)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # Adjust based on Atari preprocessing
            nn.ReLU()
        )

        # Quantum processing layer
        self.device = qml.device(backend, wires=n_qubits)
        self.quantum_params = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        # Feature compression for quantum layer
        self.quantum_input = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )

        # Output processing
        self.output_layer = nn.Sequential(
            nn.Linear(512 + n_qubits, 256),  # Classical + quantum features
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        self.qnode = qml.QNode(self._quantum_circuit, self.device, diff_method="parameter-shift")

    def _quantum_circuit(self, inputs: torch.Tensor, params: torch.Tensor) -> List[float]:
        """Simplified quantum circuit for hybrid approach."""
        # Encoding
        for i in range(self.n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            for i in range(self.n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Classical CNN feature extraction
        cnn_features = self.cnn(x)

        # Quantum processing
        quantum_input = self.quantum_input(cnn_features)

        quantum_outputs = []
        for i in range(batch_size):
            q_out = self.qnode(quantum_input[i], self.quantum_params)
            q_tensor = torch.stack([
                torch.as_tensor(val, dtype=torch.float32, device=x.device)
                for val in q_out
            ])
            quantum_outputs.append(q_tensor)

        quantum_features = torch.stack(quantum_outputs)

        # Combine classical and quantum features
        combined_features = torch.cat([cnn_features, quantum_features], dim=1)

        # Output Q-values
        q_values = self.output_layer(combined_features)

        return q_values


class QuantumReplayBuffer:
    """
    Experience replay buffer optimized for quantum RL training.
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class QuantumDQNTrainer:
    """
    Training orchestrator for Quantum DQN agents.
    Compatible with OpenAI Gym environments.
    """

    def __init__(
        self,
        env_name: str = "SpaceInvadersNoFrameskip-v4",
        model_type: str = "hybrid",  # "quantum", "hybrid"
        n_qubits: int = 6,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 100000,
        target_update: int = 10000,
        batch_size: int = 32,
        buffer_size: int = 100000,
        device: str = "cpu"
    ):
        """
        Initialize Quantum DQN trainer.

        Args:
            env_name: Gym environment name
            model_type: Type of quantum model ("quantum" or "hybrid")
            n_qubits: Number of qubits
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps for epsilon decay
            target_update: Steps between target network updates
            batch_size: Batch size for training
            buffer_size: Replay buffer size
            device: Training device
        """
        self.env_name = env_name
        self.model_type = model_type
        self.n_qubits = n_qubits
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Create environment
        self.env = gym.make(env_name)

        # Wrap with Atari preprocessing if needed
        if "NoFrameskip" in env_name:
            from gymnasium.wrappers import AtariPreprocessing, FrameStack
            self.env = AtariPreprocessing(self.env, noop_max=30, frame_skip=4,
                                        screen_size=84, terminal_on_life_loss=True,
                                        grayscale_obs=True, grayscale_newaxis=False,
                                        scale_obs=True)
            self.env = FrameStack(self.env, 4)

        self.n_actions = self.env.action_space.n

        # Create models
        if model_type == "quantum":
            self.q_network = QuantumDQN(n_qubits, self.n_actions).to(self.device)
            self.target_network = QuantumDQN(n_qubits, self.n_actions).to(self.device)
        elif model_type == "hybrid":
            self.q_network = HybridQuantumDQN(n_qubits, self.n_actions).to(self.device)
            self.target_network = HybridQuantumDQN(n_qubits, self.n_actions).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = QuantumReplayBuffer(buffer_size)

        # Training state
        self.step_count = 0
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return self.env.action_space.sample()

    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, n_episodes: int = 1000, max_steps_per_episode: int = 10000):
        """
        Train the quantum DQN agent.

        Args:
            n_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode
        """
        print(f"🚀 Training Quantum DQN ({self.model_type}) on {self.env_name}")
        print(f"📊 Model: {self.n_qubits} qubits, {self.n_actions} actions")

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0

            for step in range(max_steps_per_episode):
                # Calculate epsilon
                epsilon = max(self.epsilon_end,
                            self.epsilon_start - (self.step_count / self.epsilon_decay))

                # Select and perform action
                action = self.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store experience
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Training
                if self.step_count > 1000:  # Start training after some experience
                    loss = self.train_step()

                # Update target network
                if self.step_count % self.target_update == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                state = next_state
                total_reward += reward
                self.step_count += 1

                if done:
                    break

            self.episode_rewards.append(total_reward)

            # Print progress
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {epsilon:.3f}, Steps: {self.step_count}")

        print(f"✅ Training completed!")
        print(f"📈 Final average reward (last 100 episodes): {np.mean(self.episode_rewards[-100:]):.2f}")

        return self.episode_rewards


def create_quantum_rl_benchmark_suite():
    """
    Create a suite of quantum RL models for benchmarking against classical QRDQN.
    """

    benchmark_configs = {
        "Quantum-DQN-6q": {
            "model_type": "quantum",
            "n_qubits": 6,
            "description": "Pure quantum DQN with 6 qubits (64D feature space)"
        },
        "Quantum-DQN-8q": {
            "model_type": "quantum",
            "n_qubits": 8,
            "description": "Pure quantum DQN with 8 qubits (256D feature space)"
        },
        "Hybrid-QRL-4q": {
            "model_type": "hybrid",
            "n_qubits": 4,
            "description": "Hybrid CNN + quantum with 4 qubits"
        },
        "Hybrid-QRL-6q": {
            "model_type": "hybrid",
            "n_qubits": 6,
            "description": "Hybrid CNN + quantum with 6 qubits"
        }
    }

    print("🎮 Quantum RL Benchmark Suite for Space Invaders")
    print("=" * 60)
    print("🎯 Target: Beat classical QRDQN performance (578.00 ± 134.37)")
    print()

    for name, config in benchmark_configs.items():
        print(f"📊 {name}:")
        print(f"   Type: {config['model_type']}")
        print(f"   Qubits: {config['n_qubits']}")
        print(f"   Description: {config['description']}")
        print()

    return benchmark_configs


if __name__ == "__main__":
    # Demo the quantum RL setup
    print("🎮 Quantum Reinforcement Learning for Space Invaders")
    print("=" * 60)

    # Create benchmark suite
    configs = create_quantum_rl_benchmark_suite()

    # Example: Quick test of hybrid model
    print("🧪 Testing Hybrid Quantum RL model...")

    # Suppress warnings for demo
    warnings.filterwarnings("ignore")

    try:
        trainer = QuantumDQNTrainer(
            env_name="CartPole-v1",  # Use simpler env for demo
            model_type="hybrid",
            n_qubits=4,
            learning_rate=1e-3
        )

        print("✅ Quantum RL trainer initialized successfully!")
        print(f"🎯 Environment: {trainer.env_name}")
        print(f"⚛️  Model: {trainer.model_type} with {trainer.n_qubits} qubits")
        print(f"🎮 Actions: {trainer.n_actions}")

        # Quick training demo (few episodes)
        print("\n🚀 Running quick training demo...")
        rewards = trainer.train(n_episodes=5)
        print(f"📈 Demo rewards: {rewards}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Note: Full Space Invaders training requires more setup")

    print("\n🎯 Ready for Space Invaders quantum vs classical benchmark!")