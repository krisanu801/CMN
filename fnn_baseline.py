"""
Baseline Feedforward Neural Network (FNN) for comparison with CMN.

This is a standard 3-layer MLP that learns via external supervision.
In the CMN experiment, the external loss will be made useless (zero/random)
to demonstrate that standard networks cannot learn internal invariants.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class FeedforwardNetwork:
    """
    Standard 3-layer feedforward neural network.
    
    This serves as the baseline to compare against CMN. It learns only
    from external supervision signals, which will be made useless in
    the experiment to show it cannot form internal commitments.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] = [8, 8],
        output_dim: int = 1,
        learning_rate: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize the FNN.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer sizes
            output_dim: Output dimension
            learning_rate: Learning rate for gradient descent
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Build layer dimensions
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(layer_dims) - 1):
            # Xavier/He initialization
            w = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2.0 / layer_dims[i])
            b = np.zeros(layer_dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # History tracking
        self.history: Dict[str, List] = {
            'loss': [],
            'weight_norms': [],
            'predictions': []
        }
        
        # Cache for backpropagation
        self.activations = []
        self.pre_activations = []
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid."""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input vector
            
        Returns:
            Output prediction
        """
        self.activations = [x]
        self.pre_activations = []
        
        current = x
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            self.pre_activations.append(z)
            current = self._relu(z)
            self.activations.append(current)
        
        # Output layer with sigmoid
        z = current @ self.weights[-1] + self.biases[-1]
        self.pre_activations.append(z)
        output = self._sigmoid(z)
        self.activations.append(output)
        
        # Return scalar for single output
        return output.item() if output.size == 1 else output
    
    def compute_loss(self, y_true: float, y_pred: float) -> float:
        """
        Compute binary cross-entropy loss.
        
        In the CMN experiment, this will be made useless (constant/random).
        """
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    def backward(self, y_true: float, y_pred: float) -> None:
        """
        Backward pass (backpropagation).
        
        Args:
            y_true: True label
            y_pred: Predicted output
        """
        # Gradient of loss w.r.t. output (scalar)
        delta = y_pred - y_true
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient w.r.t. weights and biases
            # Ensure delta is 1D for outer product
            if np.isscalar(delta):
                delta_vec = np.array([delta])
            else:
                delta_vec = delta.flatten()
            
            grad_w = np.outer(self.activations[i], delta_vec)
            grad_b = delta_vec
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b
            
            # Propagate error to previous layer
            if i > 0:
                delta = (delta_vec @ self.weights[i].T) * self._relu_derivative(self.pre_activations[i-1])
                delta = delta.flatten()  # Keep as 1D array
    
    def train_step(self, x: np.ndarray, y: float) -> float:
        """
        Perform one training step.
        
        Args:
            x: Input vector
            y: Target label
            
        Returns:
            Loss value
        """
        # Forward pass
        y_pred = self.forward(x)
        
        # Compute loss
        loss = self.compute_loss(y, y_pred)
        
        # Backward pass
        self.backward(y, y_pred)
        
        # Record history
        self.history['loss'].append(loss)
        self.history['predictions'].append(y_pred)
        weight_norm = np.sqrt(sum(np.sum(w**2) for w in self.weights))
        self.history['weight_norms'].append(weight_norm)
        
        return loss
    
    def predict(self, x: np.ndarray) -> float:
        """
        Make a prediction without updating weights.
        
        Args:
            x: Input vector
            
        Returns:
            Predicted output
        """
        return self.forward(x)
    
    def get_weight_statistics(self) -> Dict:
        """Get statistics about the network weights."""
        weight_norms = [np.linalg.norm(w) for w in self.weights]
        return {
            'weight_norms': weight_norms,
            'total_norm': np.sqrt(sum(np.sum(w**2) for w in self.weights)),
            'mean_weight': np.mean([np.mean(np.abs(w)) for w in self.weights]),
            'max_weight': np.max([np.max(np.abs(w)) for w in self.weights])
        }
    
    def reset(self):
        """Reset the network to initial random state."""
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2.0 / layer_dims[i])
            b = np.zeros(layer_dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        self.history = {
            'loss': [],
            'weight_norms': [],
            'predictions': []
        }
    
    def __repr__(self) -> str:
        stats = self.get_weight_statistics()
        return (f"FNN(layers={[self.input_dim] + self.hidden_dims + [self.output_dim]}, "
                f"weight_norm={stats['total_norm']:.4f}, "
                f"steps={len(self.history['loss'])})")


if __name__ == "__main__":
    # Quick test
    print("Testing Feedforward Neural Network...")
    fnn = FeedforwardNetwork(input_dim=2, hidden_dims=[8, 8], output_dim=1, seed=42)
    print(f"Initial state: {fnn}")
    
    # Train on some dummy data
    for i in range(10):
        x = np.random.randn(2)
        y = 1.0 if x[0] > 0 else 0.0
        loss = fnn.train_step(x, y)
        if i % 5 == 0:
            print(f"Step {i}: loss={loss:.4f}, {fnn}")
    
    print("\nFNN implementation complete!")
