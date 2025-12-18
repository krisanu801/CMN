"""
Constraint-Manifold Network (CMN) Implementation

A neural network architecture with semantic closure, operating on a constrained
state space (unit hypersphere) with multi-timescale dynamics:
- Fast: State evolution (reasoning)
- Intermediate: Weight learning
- Slow: Commitment evolution (stress-induced plasticity)

Mathematical Foundation:
- State: x(t) ∈ ℝⁿ with ||x|| = 1
- Constraints: W ∈ ℝⁿˣⁿ (structural invariants)
- Commitment: κ ∈ [0,1]ⁿˣⁿ (internal normativity)
- Violation: v(t) = ||Ψ(x, W)||² where Ψ(x, W) = x - φ(Wx)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt


class ConstraintManifoldNetwork:
    """
    Constraint-Manifold Network with semantic closure.
    
    This network learns internal invariants and can detect contradictions
    independent of external supervision signals.
    """
    
    def __init__(
        self,
        n_dims: int = 2,
        alpha: float = 10.0,      # Fast timescale (state dynamics)
        eta: float = 1.0,          # Intermediate timescale (weight learning)
        gamma1: float = 0.1,       # Slow timescale (commitment increase)
        gamma2: float = 0.05,      # Slow timescale (commitment decrease)
        xi: float = 1.0,           # Permeability gate sensitivity
        dt: float = 0.01,          # Integration timestep
        seed: Optional[int] = None
    ):
        """
        Initialize the CMN.
        
        Args:
            n_dims: Dimensionality of state space
            alpha: Fast dynamics rate (state evolution)
            eta: Weight learning rate
            gamma1: Commitment increase rate (consistency)
            gamma2: Commitment decrease rate (contradiction)
            xi: Permeability gate sensitivity parameter
            dt: Integration timestep
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_dims = n_dims
        self.alpha = alpha
        self.eta = eta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.xi = xi
        self.dt = dt
        
        # State variables
        self.x = self._random_unit_vector(n_dims)  # State on unit hypersphere
        self.W = np.random.randn(n_dims, n_dims) * 0.1  # Constraint matrix
        self.kappa = np.zeros((n_dims, n_dims))  # Commitment metric [0, 1]
        
        # History tracking
        self.history: Dict[str, List] = {
            'x': [],
            'W': [],
            'kappa': [],
            'violation': [],
            'permeability': [],
            'time': []
        }
        self.time = 0.0
    
    def _random_unit_vector(self, n: int) -> np.ndarray:
        """Generate a random unit vector."""
        v = np.random.randn(n)
        return v / np.linalg.norm(v)
    
    def _project_to_sphere(self, x: np.ndarray) -> np.ndarray:
        """Project vector onto unit hypersphere."""
        norm = np.linalg.norm(x)
        if norm < 1e-10:
            # Avoid division by zero - return random unit vector
            return self._random_unit_vector(len(x))
        return x / norm
    
    def _nonlinearity(self, x: np.ndarray) -> np.ndarray:
        """Nonlinearity φ for invariant operator. Using tanh."""
        return np.tanh(x)
    
    def _nonlinearity_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of nonlinearity for gradient computation."""
        tanh_x = np.tanh(x)
        return 1 - tanh_x**2
    
    def compute_invariant_operator(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute Ψ(x, W) = x - φ(Wx).
        
        This measures the local tension/inconsistency between the state x
        and the constraints encoded in W.
        """
        return x - self._nonlinearity(W @ x)
    
    def compute_violation(self, x: Optional[np.ndarray] = None, 
                         W: Optional[np.ndarray] = None) -> float:
        """
        Compute violation signal v(t) = ||Ψ(x, W)||².
        
        This is the internal error signal - high when the state is
        inconsistent with the learned constraints.
        """
        if x is None:
            x = self.x
        if W is None:
            W = self.W
        
        psi = self.compute_invariant_operator(x, W)
        return np.sum(psi**2)
    
    def compute_permeability(self, v: float) -> float:
        """
        Compute permeability gate β(v) = exp(-ξv).
        
        Controls how much external input influences the system based on
        internal consistency. High violation → low permeability.
        """
        return np.exp(-self.xi * v)
    
    def compute_violation_gradient_x(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute ∇ₓv where v = ||Ψ(x, W)||².
        
        Using chain rule: ∇ₓv = 2Ψᵀ∇ₓΨ
        where ∇ₓΨ = I - diag(φ'(Wx))W
        """
        psi = self.compute_invariant_operator(x, W)
        Wx = W @ x
        phi_prime = self._nonlinearity_derivative(Wx)
        
        # ∇ₓΨ = I - diag(φ'(Wx))W
        grad_psi = np.eye(self.n_dims) - np.diag(phi_prime) @ W
        
        # ∇ₓv = 2Ψᵀ∇ₓΨ
        grad_v = 2 * psi @ grad_psi
        
        return grad_v
    
    def compute_violation_gradient_W(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute ∇_W v where v = ||Ψ(x, W)||².
        
        Using chain rule: ∇_W v = 2Ψᵀ∇_W Ψ
        where ∇_W Ψ = -diag(φ'(Wx))xᵀ
        """
        psi = self.compute_invariant_operator(x, W)
        Wx = W @ x
        phi_prime = self._nonlinearity_derivative(Wx)
        
        # ∇_W Ψ = -diag(φ'(Wx))xᵀ (outer product)
        # For each element: ∂Ψᵢ/∂Wⱼₖ = -φ'(Wx)ᵢ δᵢⱼ xₖ
        grad_W = -2 * np.outer(psi * phi_prime, x)
        
        return grad_W
    
    def state_dynamics(self, x: np.ndarray, W: np.ndarray, 
                      u: np.ndarray, beta: float) -> np.ndarray:
        """
        Compute ẋ = -α∇ₓv + β(v)u, then project to sphere.
        
        Fast dynamics: State evolves to minimize violation while being
        perturbed by gated external input.
        """
        grad_v_x = self.compute_violation_gradient_x(x, W)
        dx_dt = -self.alpha * grad_v_x + beta * u
        
        return dx_dt
    
    def weight_dynamics(self, x: np.ndarray, W: np.ndarray, 
                       kappa: np.ndarray) -> np.ndarray:
        """
        Compute Ẇ = -η(1 - κ)∇_W v.
        
        Intermediate dynamics: Weights update to reduce violation, but only
        where commitment κ is low (plasticity).
        """
        grad_v_W = self.compute_violation_gradient_W(x, W)
        dW_dt = -self.eta * (1 - kappa) * grad_v_W
        
        return dW_dt
    
    def commitment_dynamics(self, kappa: np.ndarray, v: float) -> np.ndarray:
        """
        Compute κ̇ = γ₁(1 - κ)exp(-v) - γ₂κv.
        
        Slow dynamics: Commitment increases with consistency (low v) and
        decreases with persistent contradiction (high v).
        This implements stress-induced plasticity (falsifiability).
        """
        # Increase term: consistency strengthens commitment
        increase = self.gamma1 * (1 - kappa) * np.exp(-v)
        
        # Decrease term: contradiction weakens commitment
        decrease = self.gamma2 * kappa * v
        
        dkappa_dt = increase - decrease
        
        return dkappa_dt
    
    def step(self, u: np.ndarray) -> Dict[str, float]:
        """
        Perform one integration step with multi-timescale dynamics.
        
        Args:
            u: External input perturbation
            
        Returns:
            Dictionary of current metrics
        """
        # Compute current violation and permeability
        v = self.compute_violation()
        beta = self.compute_permeability(v)
        
        # Update state (fast dynamics)
        dx_dt = self.state_dynamics(self.x, self.W, u, beta)
        x_new = self.x + self.dt * dx_dt
        self.x = self._project_to_sphere(x_new)  # Maintain ||x|| = 1
        
        # Update weights (intermediate dynamics)
        dW_dt = self.weight_dynamics(self.x, self.W, self.kappa)
        self.W = self.W + self.dt * dW_dt
        
        # Update commitment (slow dynamics)
        dkappa_dt = self.commitment_dynamics(self.kappa, v)
        self.kappa = np.clip(self.kappa + self.dt * dkappa_dt, 0, 1)
        
        # Update time
        self.time += self.dt
        
        # Record history
        self.history['x'].append(self.x.copy())
        self.history['W'].append(self.W.copy())
        self.history['kappa'].append(self.kappa.copy())
        self.history['violation'].append(v)
        self.history['permeability'].append(beta)
        self.history['time'].append(self.time)
        
        return {
            'violation': v,
            'permeability': beta,
            'commitment_mean': np.mean(self.kappa),
            'time': self.time
        }
    
    def reset(self):
        """Reset the network to initial random state."""
        self.x = self._random_unit_vector(self.n_dims)
        self.W = np.random.randn(self.n_dims, self.n_dims) * 0.1
        self.kappa = np.zeros((self.n_dims, self.n_dims))
        self.time = 0.0
        self.history = {
            'x': [],
            'W': [],
            'kappa': [],
            'violation': [],
            'permeability': [],
            'time': []
        }
    
    def get_state(self) -> Dict:
        """Get current state of the network."""
        return {
            'x': self.x.copy(),
            'W': self.W.copy(),
            'kappa': self.kappa.copy(),
            'violation': self.compute_violation(),
            'permeability': self.compute_permeability(self.compute_violation()),
            'time': self.time
        }
    
    def __repr__(self) -> str:
        v = self.compute_violation()
        beta = self.compute_permeability(v)
        return (f"CMN(n_dims={self.n_dims}, t={self.time:.2f}, "
                f"v={v:.4f}, β={beta:.4f}, "
                f"κ_mean={np.mean(self.kappa):.4f})")


if __name__ == "__main__":
    # Quick test
    print("Testing Constraint-Manifold Network...")
    cmn = ConstraintManifoldNetwork(n_dims=2, seed=42)
    print(f"Initial state: {cmn}")
    
    # Simulate a few steps with random input
    for i in range(10):
        u = np.random.randn(2) * 0.1
        metrics = cmn.step(u)
        if i % 5 == 0:
            print(f"Step {i}: {cmn}")
    
    print("\nCMN implementation complete!")
