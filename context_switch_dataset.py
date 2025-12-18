"""
Context Switching Dataset

Generates a continuous input stream that switches rules mid-stream without warning.
This tests the H-CMN's ability to detect the context change and adapt via L2 modulation.

Dataset Structure:
- Steps 0-499: Rule A (Red→Square, Blue→Circle)
- Step 500: ABRUPT SWITCH (no warning)
- Steps 500-999: Rule B (Red→Circle, Blue→Square)

The H-CMN should:
1. Learn Rule A during steps 0-499
2. Experience violation spike at step 500
3. Adapt via L2 context shift
4. Settle into Rule B by step 600
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ContextSample:
    """A single sample from the context switching stream."""
    step: int
    color: str  # "Red" or "Blue"
    shape: str  # "Square" or "Circle"
    encoding: np.ndarray  # [color_val, shape_val]
    rule: str  # "Rule A" or "Rule B"
    is_switch_point: bool  # True if this is step 500
    
    def __repr__(self) -> str:
        switch_marker = " ← SWITCH!" if self.is_switch_point else ""
        return f"Step {self.step}: {self.color} {self.shape} -> {self.encoding} ({self.rule}){switch_marker}"


class ContextSwitchDataset:
    """
    Context switching dataset for testing hierarchical semantic closure.
    
    Generates a stream that abruptly switches rules at a specified point,
    testing whether the H-CMN can detect and adapt to the context change.
    """
    
    def __init__(
        self,
        n_steps: int = 1000,
        switch_point: int = 500,
        seed: Optional[int] = None
    ):
        """
        Initialize the context switching dataset.
        
        Args:
            n_steps: Total number of steps in the stream
            switch_point: Step at which to switch rules
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_steps = n_steps
        self.switch_point = switch_point
        
        # Generate the full stream
        self.stream = self._generate_stream()
    
    def _generate_rule_a_sample(self, step: int) -> ContextSample:
        """
        Generate a sample from Rule A: Red→Square, Blue→Circle.
        """
        if np.random.rand() > 0.5:
            # Red Square
            return ContextSample(
                step=step,
                color="Red",
                shape="Square",
                encoding=np.array([1.0, 1.0]),
                rule="Rule A",
                is_switch_point=False
            )
        else:
            # Blue Circle
            return ContextSample(
                step=step,
                color="Blue",
                shape="Circle",
                encoding=np.array([-1.0, -1.0]),
                rule="Rule A",
                is_switch_point=False
            )
    
    def _generate_rule_b_sample(self, step: int, is_switch: bool = False) -> ContextSample:
        """
        Generate a sample from Rule B: Red→Circle, Blue→Square.
        """
        if np.random.rand() > 0.5:
            # Red Circle (CONTRADICTS Rule A!)
            return ContextSample(
                step=step,
                color="Red",
                shape="Circle",
                encoding=np.array([1.0, -1.0]),
                rule="Rule B",
                is_switch_point=is_switch
            )
        else:
            # Blue Square (CONTRADICTS Rule A!)
            return ContextSample(
                step=step,
                color="Blue",
                shape="Square",
                encoding=np.array([-1.0, 1.0]),
                rule="Rule B",
                is_switch_point=is_switch
            )
    
    def _generate_stream(self) -> List[ContextSample]:
        """
        Generate the complete input stream with rule switch.
        """
        stream = []
        
        for step in range(self.n_steps):
            if step < self.switch_point:
                # Rule A
                sample = self._generate_rule_a_sample(step)
            elif step == self.switch_point:
                # SWITCH POINT!
                sample = self._generate_rule_b_sample(step, is_switch=True)
            else:
                # Rule B
                sample = self._generate_rule_b_sample(step, is_switch=False)
            
            stream.append(sample)
        
        return stream
    
    def get_stream(self) -> List[ContextSample]:
        """Get the complete input stream."""
        return self.stream
    
    def get_sample(self, step: int) -> ContextSample:
        """Get a specific sample by step number."""
        if step < 0 or step >= self.n_steps:
            raise ValueError(f"Step {step} out of range [0, {self.n_steps})")
        return self.stream[step]
    
    def get_switch_point(self) -> int:
        """Get the step number where rules switch."""
        return self.switch_point
    
    def get_pre_switch_samples(self) -> List[ContextSample]:
        """Get all samples before the switch (Rule A)."""
        return self.stream[:self.switch_point]
    
    def get_post_switch_samples(self) -> List[ContextSample]:
        """Get all samples after the switch (Rule B)."""
        return self.stream[self.switch_point:]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        rule_a_count = sum(1 for s in self.stream if s.rule == "Rule A")
        rule_b_count = sum(1 for s in self.stream if s.rule == "Rule B")
        
        return {
            'n_steps': self.n_steps,
            'switch_point': self.switch_point,
            'rule_a_steps': rule_a_count,
            'rule_b_steps': rule_b_count,
            'rule_a_ratio': rule_a_count / self.n_steps,
            'rule_b_ratio': rule_b_count / self.n_steps
        }
    
    def visualize_stream(self, window_size: int = 20):
        """
        Visualize the stream structure around the switch point.
        
        Args:
            window_size: Number of samples to show before/after switch
        """
        print("=" * 70)
        print("CONTEXT SWITCHING DATASET STRUCTURE")
        print("=" * 70)
        
        stats = self.get_statistics()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n{'Rule A (Pre-Switch)':-^70}")
        print("Red → Square: [1, 1]")
        print("Blue → Circle: [-1, -1]")
        print(f"\nSamples near switch point (steps {self.switch_point - window_size} to {self.switch_point - 1}):")
        for i in range(max(0, self.switch_point - window_size), self.switch_point):
            if i % 5 == 0 or i >= self.switch_point - 5:
                print(f"  {self.stream[i]}")
        
        print(f"\n{'SWITCH POINT (Step {self.switch_point})':-^70}")
        print("Rules change abruptly - NO WARNING!")
        print(f"  {self.stream[self.switch_point]}")
        
        print(f"\n{'Rule B (Post-Switch)':-^70}")
        print("Red → Circle: [1, -1]  ← CONTRADICTS Rule A!")
        print("Blue → Square: [-1, 1]  ← CONTRADICTS Rule A!")
        print(f"\nSamples after switch (steps {self.switch_point + 1} to {self.switch_point + window_size}):")
        for i in range(self.switch_point + 1, min(self.n_steps, self.switch_point + window_size + 1)):
            if i % 5 == 0 or i <= self.switch_point + 5:
                print(f"  {self.stream[i]}")
        
        print("\n" + "=" * 70)
        print("H-CMN Expected Behavior:")
        print("  1. Steps 0-499: Learn Rule A (v₁ → 0, x₂ stable)")
        print("  2. Step 500: Violation spike (v₁ >> 0)")
        print("  3. Steps 500-600: L2 adapts (x₂ transitions, W₁ morphs)")
        print("  4. Steps 600+: Settle into Rule B (v₁ → 0, x₂ stable)")
        print("=" * 70)


if __name__ == "__main__":
    from typing import Optional
    
    print("Testing Context Switching Dataset...\n")
    
    dataset = ContextSwitchDataset(
        n_steps=1000,
        switch_point=500,
        seed=42
    )
    
    dataset.visualize_stream(window_size=10)
    
    print("\n\nTesting sample access...")
    print(f"Sample at step 499 (last Rule A): {dataset.get_sample(499)}")
    print(f"Sample at step 500 (switch): {dataset.get_sample(500)}")
    print(f"Sample at step 501 (first Rule B): {dataset.get_sample(501)}")
    
    print("\nContext Switching Dataset implementation complete!")
