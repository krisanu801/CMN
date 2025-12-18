"""
CMN Network with additional methods for testing semantic closure.
"""

import numpy as np
from typing import Dict

def compute_violation_with_input(cmn, u: np.ndarray) -> float:
    """
    Compute what the violation WOULD BE if we set x to the input u.
    
    This is used in Phase 2 to test semantic closure:
    - We project u onto the unit sphere to get a candidate state
    - We compute the violation of this state against the learned constraints W
    - We DON'T update the network
    
    Args:
        cmn: CMN network instance
        u: Input vector (will be projected to unit sphere)
        
    Returns:
        Violation signal v for the input
    """
    # Project input to unit sphere
    x_candidate = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-10 else cmn._random_unit_vector(len(u))
    
    # Compute violation with this candidate state and the LEARNED weights
    v = cmn.compute_violation(x=x_candidate, W=cmn.W)
    
    return v


def freeze_commitments(cmn):
    """
    Freeze all commitments to maximum (κ = 1.0).
    
    This prevents weight updates in Phase 2, ensuring the network
    maintains its learned invariants and can detect contradictions.
    """
    cmn.kappa = np.ones_like(cmn.kappa)
    print(f"  [CMN] Commitments frozen at κ = 1.0 (weights locked)")
