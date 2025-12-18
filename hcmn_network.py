"""
Hierarchical Constraint-Manifold Network (H-CMN) Implementation

A two-level neural architecture with contextual governance, where Level 2 
monitors Level 1's failure states and intervenes by modulating constraints 
in real-time.

Key Innovation: Top-Down Constraint Modulation
- L1 (Sensory): Processes input with dynamic constraints W₁(x₂)
- L2 (Executive): Monitors L1's violation v₁ and adjusts its context x₂
- Empathy Gradient: ∇_{x₂}v₁ tells L2 how to help L1 succeed

Mathematical Foundation:
- L1 State: x₁(t) ∈ ℝⁿ¹ with ||x₁|| = 1
- L2 State: x₂(t) ∈ ℝⁿ² with ||x₂|| = 1
- Dynamic Weights: W₁(x₂) = Σᵢ Θ[:,:,i] · x₂[i] (tensor contraction)
- Coupled Violation: v₂ = v₂_local + λv₁ (empathy)
- Empathy Gradient: ∇_{x₂}v₁ (how L2 helps L1)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt


class HierarchicalCMN:
    """
    Hierarchical Constraint-Manifold Network with contextual governance.
    
    Two-level coupled dynamical system where L2 (executive) monitors L1 
    (sensory) and modulates its constraints to resolve contradictions.
    """
    
    def __init__(
        self,
        n1: int = 2,              # L1 dimensionality
        n2: int = 4,              # L2 dimensionality
        alpha1: float = 10.0,     # L1 fast dynamics rate
        alpha2: float = 5.0,      # L2 fast dynamics rate
        eta1: float = 1.0,        # L1 weight learning rate
        eta2: float = 0.5,        # L2 weight learning rate
        gamma1_l1: float = 0.1,   # L1 commitment increase
        gamma2_l1: float = 0.05,  # L1 commitment decrease
        gamma1_l2: float = 0.1,   # L2 commitment increase
        gamma2_l2: float = 0.05,  # L2 commitment decrease
        lambda_coupling: float = 1.0,  # Empathy coupling strength
        xi1: float = 1.0,         # L1 permeability sensitivity
        xi2: float = 1.0,         # L2 permeability sensitivity
        dt: float = 0.01,         # Integration timestep
        seed: Optional[int] = None
    ):
        """
        Initialize the H-CMN.
        
        Args:
            n1: L1 state dimensionality
            n2: L2 state dimensionality (should be >= n1 for sufficient capacity)
            alpha1, alpha2: Fast dynamics rates
            eta1, eta2: Weight learning rates
            gamma1_l1, gamma2_l1: L1 commitment dynamics
            gamma1_l2, gamma2_l2: L2 commitment dynamics
            lambda_coupling: Empathy coupling strength (how much L2 cares about L1)
            xi1, xi2: Permeability gate sensitivities
            dt: Integration timestep
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n1 = n1
        self.n2 = n2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma1_l1 = gamma1_l1
        self.gamma2_l1 = gamma2_l1
        self.gamma1_l2 = gamma1_l2
        self.gamma2_l2 = gamma2_l2
        self.lambda_coupling = lambda_coupling
        self.xi1 = xi1
        self.xi2 = xi2
        self.dt = dt
        
        # Level 1 state variables
        self.x1 = self._random_unit_vector(n1)
        self.kappa1 = np.zeros((n1, n1))
        
        # Level 2 state variables
        self.x2 = self._random_unit_vector(n2)
        self.W2 = np.random.randn(n2, n2) * 0.1  # L2 has static weights
        self.kappa2 = np.zeros((n2, n2))
        
        # Hyper-weight tensor Θ ∈ ℝⁿ¹ˣⁿ¹ˣⁿ²
        # This encodes all possible "rule sets" that L2 can select
        self.Theta = self._initialize_hyper_weights()
        
        # History tracking
        self.history: Dict[str, List] = {
            'x1': [],
            'x2': [],
            'W1': [],
            'W2': [],
            'kappa1': [],
            'kappa2': [],
            'v1': [],
            'v2': [],
            'v2_local': [],
            'beta1': [],
            'beta2': [],
            'empathy_gradient_norm': [],
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
            return self._random_unit_vector(len(x))
        return x / norm
    
    def _initialize_hyper_weights(self) -> np.ndarray:
        """
        Initialize hyper-weight tensor Θ ∈ ℝⁿ¹ˣⁿ¹ˣⁿ².
        
        This tensor encodes multiple possible "rule sets" that L2 can select.
        Each slice Θ[:,:,i] represents a different constraint configuration.
        
        We initialize with EXPLICIT rule encoding:
        - Context 0: Rule A (Red→Square, Blue→Circle)
        - Context 1: Rule B (Red→Circle, Blue→Square)
        - Contexts 2-3: Neutral/interpolation states
        
        This creates a clear distinction between incompatible rule sets.
        """
        Theta = np.zeros((self.n1, self.n1, self.n2))
        
        if self.n1 == 2 and self.n2 >= 2:
            # Context 0: Rule A (positive correlation)
            # Encodes: x[0]=x[1] (Red→Square, Blue→Circle)
            Theta[:, :, 0] = np.array([
                [1.5, 1.5],   # Strong positive correlation
                [1.5, 1.5]
            ])
            
            # Context 1: Rule B (negative correlation)
            # Encodes: x[0]=-x[1] (Red→Circle, Blue→Square)
            Theta[:, :, 1] = np.array([
                [1.5, -1.5],  # Strong negative correlation
                [-1.5, 1.5]
            ])
            
            # Contexts 2-3: Neutral/transition states
            if self.n2 > 2:
                Theta[:, :, 2] = np.random.randn(self.n1, self.n1) * 0.1
            if self.n2 > 3:
                Theta[:, :, 3] = np.random.randn(self.n1, self.n1) * 0.1
        else:
            # General case: diverse patterns
            for i in range(self.n2):
                if i % 2 == 0:
                    Theta[:, :, i] = np.eye(self.n1) + np.random.randn(self.n1, self.n1) * 0.1
                else:
                    Theta[:, :, i] = -np.eye(self.n1) + np.random.randn(self.n1, self.n1) * 0.1
            Theta = Theta * 0.5
        
        return Theta
    
    def _nonlinearity(self, x: np.ndarray) -> np.ndarray:
        """Nonlinearity φ for invariant operator."""
        return np.tanh(x)
    
    def _nonlinearity_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of nonlinearity."""
        tanh_x = np.tanh(x)
        return 1 - tanh_x**2
    
    def compute_W1(self, x2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute dynamic weight matrix W₁(x₂) via tensor contraction.
        
        W₁(x₂) = Σᵢ Θ[:,:,i] · x₂[i]
        
        This is the key mechanism: L2's state x₂ selects which "rule set" 
        is active for L1.
        
        Args:
            x2: L2 state (defaults to self.x2)
            
        Returns:
            W₁ ∈ ℝⁿ¹ˣⁿ¹
        """
        if x2 is None:
            x2 = self.x2
        
        # Tensor contraction: W1[i,j] = Σₖ Θ[i,j,k] · x₂[k]
        W1 = np.einsum('ijk,k->ij', self.Theta, x2)
        
        return W1
    
    def compute_invariant_operator_l1(self, x1: np.ndarray, W1: np.ndarray) -> np.ndarray:
        """
        Compute L1 invariant operator Ψ₁(x₁, W₁) = x₁ - φ(W₁x₁).
        """
        return x1 - self._nonlinearity(W1 @ x1)
    
    def compute_invariant_operator_l2(self, x2: np.ndarray, W2: np.ndarray) -> np.ndarray:
        """
        Compute L2 invariant operator Ψ₂(x₂, W₂) = x₂ - φ(W₂x₂).
        """
        return x2 - self._nonlinearity(W2 @ x2)
    
    def compute_v1(self, x1: Optional[np.ndarray] = None, 
                   x2: Optional[np.ndarray] = None) -> float:
        """
        Compute L1 violation v₁ = ||Ψ₁(x₁, W₁(x₂))||².
        
        This measures how well L1's state satisfies the constraints
        imposed by L2's current context.
        """
        if x1 is None:
            x1 = self.x1
        if x2 is None:
            x2 = self.x2
        
        W1 = self.compute_W1(x2)
        psi1 = self.compute_invariant_operator_l1(x1, W1)
        return np.sum(psi1**2)
    
    def compute_input_constraint_mismatch(self, u: np.ndarray, 
                                         x2: Optional[np.ndarray] = None) -> float:
        """
        Compute direct mismatch between input u and constraints W₁(x₂).
        
        This measures the INCOMPATIBILITY between the input and learned constraints.
        
        Key insight: If W₁ encodes a rule like "dimension 0 and 1 should be correlated",
        then W₁u will amplify inputs that follow the rule and suppress those that don't.
        
        We measure: -u^T(W₁u) / ||u||²
        
        This is NEGATIVE when u and W₁u point in opposite directions (contradiction!)
        and POSITIVE when they align (consistent).
        
        For example:
        - Rule A: W₁ has positive off-diagonal (correlation)
        - Input [1, 1]: W₁[1,1] gives [positive, positive] → alignment → negative mismatch
        - Input [1, -1]: W₁[1,-1] gives [small, small] → misalignment → positive mismatch
        
        Args:
            u: Input vector
            x2: L2 context state
            
        Returns:
            Mismatch value (large positive when input contradicts constraints)
        """
        if x2 is None:
            x2 = self.x2
        
        W1 = self.compute_W1(x2)
        
        # Compute how W₁ transforms the input
        W1u = W1 @ u
        
        # Measure alignment: negative dot product means contradiction
        # We want LARGE POSITIVE values for contradictions
        alignment = np.dot(u, W1u)
        u_norm_sq = np.dot(u, u)
        
        # Normalized alignment: ranges from -||W₁|| to +||W₁||
        # For contradictions, this will be negative or small
        # We flip and shift to make contradictions give large positive values
        if u_norm_sq > 1e-10:
            normalized_alignment = alignment / u_norm_sq
        else:
            normalized_alignment = 0.0
        
        # Mismatch: large when alignment is negative (contradiction)
        # Scale by W1 norm to make it significant
        W1_norm = np.linalg.norm(W1)
        mismatch = (W1_norm - normalized_alignment) ** 2
        
        return mismatch
    
    def compute_v2_local(self, x2: Optional[np.ndarray] = None) -> float:
        """
        Compute L2's local violation v₂_local = ||Ψ₂(x₂, W₂)||².
        
        This is L2's own internal consistency.
        """
        if x2 is None:
            x2 = self.x2
        
        psi2 = self.compute_invariant_operator_l2(x2, self.W2)
        return np.sum(psi2**2)
    
    def compute_v2(self, x1: Optional[np.ndarray] = None,
                   x2: Optional[np.ndarray] = None) -> float:
        """
        Compute coupled violation v₂ = v₂_local + λv₁.
        
        This is the EMPATHY mechanism: L2 cannot be at peace if L1 is struggling.
        The coupling coefficient λ determines how much L2 "cares" about L1.
        """
        v1 = self.compute_v1(x1, x2)
        v2_local = self.compute_v2_local(x2)
        
        return v2_local + self.lambda_coupling * v1
    
    def compute_permeability(self, v: float, xi: float) -> float:
        """Compute permeability gate β(v) = exp(-ξv)."""
        return np.exp(-xi * v)
    
    def compute_violation_gradient_x1(self, x1: np.ndarray, W1: np.ndarray) -> np.ndarray:
        """
        Compute ∇_{x₁}v₁ where v₁ = ||Ψ₁(x₁, W₁)||².
        
        Standard gradient for L1 state dynamics.
        """
        psi1 = self.compute_invariant_operator_l1(x1, W1)
        W1x1 = W1 @ x1
        phi_prime = self._nonlinearity_derivative(W1x1)
        
        # ∇_{x₁}Ψ₁ = I - diag(φ'(W₁x₁))W₁
        grad_psi1 = np.eye(self.n1) - np.diag(phi_prime) @ W1
        
        # ∇_{x₁}v₁ = 2Ψ₁ᵀ∇_{x₁}Ψ₁
        grad_v1 = 2 * psi1 @ grad_psi1
        
        return grad_v1
    
    def compute_violation_gradient_x2_local(self, x2: np.ndarray) -> np.ndarray:
        """
        Compute ∇_{x₂}v₂_local where v₂_local = ||Ψ₂(x₂, W₂)||².
        
        Standard gradient for L2's own consistency.
        """
        psi2 = self.compute_invariant_operator_l2(x2, self.W2)
        W2x2 = self.W2 @ x2
        phi_prime = self._nonlinearity_derivative(W2x2)
        
        # ∇_{x₂}Ψ₂ = I - diag(φ'(W₂x₂))W₂
        grad_psi2 = np.eye(self.n2) - np.diag(phi_prime) @ self.W2
        
        # ∇_{x₂}v₂_local = 2Ψ₂ᵀ∇_{x₂}Ψ₂
        grad_v2_local = 2 * psi2 @ grad_psi2
        
        return grad_v2_local
    
    def compute_empathy_gradient(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Compute empathy gradient ∇_{x₂}v₁.
        
        This is the KEY INNOVATION: it tells L2 how to change its context x₂
        to reduce L1's violation v₁.
        
        Mathematical derivation:
        v₁ = ||Ψ₁(x₁, W₁(x₂))||²
        
        Chain rule:
        ∇_{x₂}v₁ = 2Ψ₁ᵀ · ∇_{x₂}Ψ₁
        
        where:
        ∇_{x₂}Ψ₁ = -diag(φ'(W₁x₁)) · (∇_{x₂}W₁) · x₁
        
        and:
        ∇_{x₂}W₁ is given by the hyper-weight tensor Θ
        For component k: ∂W₁/∂x₂[k] = Θ[:,:,k]
        
        Returns:
            ∇_{x₂}v₁ ∈ ℝⁿ²
        """
        W1 = self.compute_W1(x2)
        psi1 = self.compute_invariant_operator_l1(x1, W1)
        W1x1 = W1 @ x1
        phi_prime = self._nonlinearity_derivative(W1x1)
        
        # Compute ∇_{x₂}v₁ component by component
        grad_v1_x2 = np.zeros(self.n2)
        
        for k in range(self.n2):
            # ∂W₁/∂x₂[k] = Θ[:,:,k]
            dW1_dx2k = self.Theta[:, :, k]
            
            # ∂Ψ₁/∂x₂[k] = -diag(φ'(W₁x₁)) · Θ[:,:,k] · x₁
            dPsi1_dx2k = -np.diag(phi_prime) @ dW1_dx2k @ x1
            
            # ∂v₁/∂x₂[k] = 2Ψ₁ᵀ · ∂Ψ₁/∂x₂[k]
            grad_v1_x2[k] = 2 * np.dot(psi1, dPsi1_dx2k)
        
        return grad_v1_x2
    
    def l1_dynamics(self, x1: np.ndarray, x2: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute L1 state dynamics ẋ₁ = -α₁∇_{x₁}v₁ + β₁(v₁)u.
        
        L1 evolves to minimize its violation while being perturbed by input.
        """
        W1 = self.compute_W1(x2)
        v1 = self.compute_v1(x1, x2)
        beta1 = self.compute_permeability(v1, self.xi1)
        
        grad_v1_x1 = self.compute_violation_gradient_x1(x1, W1)
        dx1_dt = -self.alpha1 * grad_v1_x1 + beta1 * u
        
        return dx1_dt
    
    def l2_dynamics(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Compute L2 state dynamics ẋ₂ = -α₂∇_{x₂}v₂.
        
        L2 evolves to minimize its COUPLED violation v₂ = v₂_local + λv₁.
        This includes the empathy gradient ∇_{x₂}v₁.
        """
        # Local gradient (L2's own consistency)
        grad_v2_local = self.compute_violation_gradient_x2_local(x2)
        
        # Empathy gradient (how to help L1)
        grad_v1_x2 = self.compute_empathy_gradient(x1, x2)
        
        # Combined gradient
        grad_v2 = grad_v2_local + self.lambda_coupling * grad_v1_x2
        
        dx2_dt = -self.alpha2 * grad_v2
        
        return dx2_dt
    
    def step(self, u: np.ndarray) -> Dict[str, float]:
        """
        Perform one integration step with bidirectional coupling.
        
        Args:
            u: External input to L1
            
        Returns:
            Dictionary of current metrics
        """
        # Compute current violations
        v1 = self.compute_v1()
        v2_local = self.compute_v2_local()
        v2 = self.compute_v2()
        
        # Compute permeabilities
        beta1 = self.compute_permeability(v1, self.xi1)
        beta2 = self.compute_permeability(v2, self.xi2)
        
        # Compute empathy gradient norm (for monitoring)
        empathy_grad = self.compute_empathy_gradient(self.x1, self.x2)
        empathy_grad_norm = np.linalg.norm(empathy_grad)
        
        # Update L1 state (fast dynamics)
        dx1_dt = self.l1_dynamics(self.x1, self.x2, u)
        x1_new = self.x1 + self.dt * dx1_dt
        self.x1 = self._project_to_sphere(x1_new)
        
        # Update L2 state (fast dynamics)
        dx2_dt = self.l2_dynamics(self.x1, self.x2)
        x2_new = self.x2 + self.dt * dx2_dt
        self.x2 = self._project_to_sphere(x2_new)
        
        # Update L2 weights (intermediate dynamics)
        # L1 weights are dynamic, so no update needed
        W1 = self.compute_W1()
        psi2 = self.compute_invariant_operator_l2(self.x2, self.W2)
        W2x2 = self.W2 @ self.x2
        phi_prime = self._nonlinearity_derivative(W2x2)
        grad_W2 = -2 * np.outer(psi2 * phi_prime, self.x2)
        dW2_dt = -self.eta2 * (1 - self.kappa2) * grad_W2
        self.W2 = self.W2 + self.dt * dW2_dt
        
        # Update commitments (slow dynamics)
        # L1 commitment
        dkappa1_dt = (self.gamma1_l1 * (1 - self.kappa1) * np.exp(-v1) - 
                      self.gamma2_l1 * self.kappa1 * v1)
        self.kappa1 = np.clip(self.kappa1 + self.dt * dkappa1_dt, 0, 1)
        
        # L2 commitment
        dkappa2_dt = (self.gamma1_l2 * (1 - self.kappa2) * np.exp(-v2) - 
                      self.gamma2_l2 * self.kappa2 * v2)
        self.kappa2 = np.clip(self.kappa2 + self.dt * dkappa2_dt, 0, 1)
        
        # Update time
        self.time += self.dt
        
        # Record history
        self.history['x1'].append(self.x1.copy())
        self.history['x2'].append(self.x2.copy())
        self.history['W1'].append(W1.copy())
        self.history['W2'].append(self.W2.copy())
        self.history['kappa1'].append(self.kappa1.copy())
        self.history['kappa2'].append(self.kappa2.copy())
        self.history['v1'].append(v1)
        self.history['v2'].append(v2)
        self.history['v2_local'].append(v2_local)
        self.history['beta1'].append(beta1)
        self.history['beta2'].append(beta2)
        self.history['empathy_gradient_norm'].append(empathy_grad_norm)
        self.history['time'].append(self.time)
        
        return {
            'v1': v1,
            'v2': v2,
            'v2_local': v2_local,
            'beta1': beta1,
            'beta2': beta2,
            'empathy_gradient_norm': empathy_grad_norm,
            'time': self.time
        }
    
    def get_state(self) -> Dict:
        """Get current state of the H-CMN."""
        return {
            'x1': self.x1.copy(),
            'x2': self.x2.copy(),
            'W1': self.compute_W1(),
            'W2': self.W2.copy(),
            'kappa1': self.kappa1.copy(),
            'kappa2': self.kappa2.copy(),
            'v1': self.compute_v1(),
            'v2': self.compute_v2(),
            'v2_local': self.compute_v2_local(),
            'time': self.time
        }
    
    def __repr__(self) -> str:
        v1 = self.compute_v1()
        v2 = self.compute_v2()
        return (f"H-CMN(n1={self.n1}, n2={self.n2}, t={self.time:.2f}, "
                f"v1={v1:.4f}, v2={v2:.4f})")


if __name__ == "__main__":
    # Quick test
    print("Testing Hierarchical Constraint-Manifold Network...")
    hcmn = HierarchicalCMN(n1=2, n2=4, seed=42)
    print(f"Initial state: {hcmn}")
    print(f"  L1 state x1: {hcmn.x1}")
    print(f"  L2 state x2: {hcmn.x2}")
    print(f"  Dynamic W1(x2):\n{hcmn.compute_W1()}")
    
    # Simulate a few steps
    for i in range(10):
        u = np.random.randn(2) * 0.1
        metrics = hcmn.step(u)
        if i % 5 == 0:
            print(f"\nStep {i}: {hcmn}")
            print(f"  Empathy gradient norm: {metrics['empathy_gradient_norm']:.4f}")
    
    print("\nH-CMN implementation complete!")
