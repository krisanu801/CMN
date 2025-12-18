"""
Color-Shape Invariant Set (CSIS) Dataset Generator

Generates the dataset for testing semantic closure in CMN vs FNN.

Dataset Structure:
- Phase 1 (Correlation): 90% of data follows Rule 1
  * Red → Square: encoded as [1, 1]
  * Blue → Circle: encoded as [-1, -1]
  
- Phase 2 (Paradox): 10% of data violates Rule 1
  * Red → Circle: encoded as [1, -1]
  * Blue → Square: encoded as [-1, 1]

External labels are made useless (constant or random) to force
internal learning in CMN and demonstrate FNN's inability to learn
without external supervision.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CSISample:
    """A single sample from the CSIS dataset."""
    color: str  # "Red" or "Blue"
    shape: str  # "Square" or "Circle"
    encoding: np.ndarray  # [color_val, shape_val]
    label: float  # External label (useless in experiment)
    phase: int  # 1 = correlation, 2 = paradox
    
    def __repr__(self) -> str:
        return f"{self.color} {self.shape} -> {self.encoding} (Phase {self.phase})"


class CSISDataset:
    """
    Color-Shape Invariant Set dataset generator.
    
    This dataset is designed to test whether a network can form and
    enforce internal commitments independent of external labels.
    """
    
    def __init__(
        self,
        n_correlation: int = 180,  # 90% correlation samples
        n_paradox: int = 20,       # 10% paradox samples
        label_mode: str = "constant",  # "constant", "random", or "zero"
        seed: Optional[int] = None
    ):
        """
        Initialize the CSIS dataset.
        
        Args:
            n_correlation: Number of correlation phase samples
            n_paradox: Number of paradox phase samples
            label_mode: How to generate useless external labels
                - "constant": All labels are 1.0
                - "random": Random labels in [0, 1]
                - "zero": All labels are 0.0
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_correlation = n_correlation
        self.n_paradox = n_paradox
        self.label_mode = label_mode
        
        # Generate datasets
        self.correlation_data = self._generate_correlation_phase()
        self.paradox_data = self._generate_paradox_phase()
        self.all_data = self.correlation_data + self.paradox_data
    
    def _generate_useless_label(self) -> float:
        """Generate a useless external label."""
        if self.label_mode == "constant":
            return 1.0
        elif self.label_mode == "random":
            return np.random.rand()
        elif self.label_mode == "zero":
            return 0.0
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")
    
    def _generate_correlation_phase(self) -> List[CSISample]:
        """
        Generate Phase 1 data: Correlation (Rule 1).
        
        Rule 1:
        - Red → Square: [1, 1]
        - Blue → Circle: [-1, -1]
        """
        samples = []
        
        for i in range(self.n_correlation):
            if i % 2 == 0:
                # Red Square
                sample = CSISample(
                    color="Red",
                    shape="Square",
                    encoding=np.array([1.0, 1.0]),
                    label=self._generate_useless_label(),
                    phase=1
                )
            else:
                # Blue Circle
                sample = CSISample(
                    color="Blue",
                    shape="Circle",
                    encoding=np.array([-1.0, -1.0]),
                    label=self._generate_useless_label(),
                    phase=1
                )
            samples.append(sample)
        
        return samples
    
    def _generate_paradox_phase(self) -> List[CSISample]:
        """
        Generate Phase 2 data: Paradox (violates Rule 1).
        
        Rule 2 (opposite pairings):
        - Red → Circle: [1, -1]
        - Blue → Square: [-1, 1]
        """
        samples = []
        
        for i in range(self.n_paradox):
            if i % 2 == 0:
                # Red Circle (PARADOX!)
                sample = CSISample(
                    color="Red",
                    shape="Circle",
                    encoding=np.array([1.0, -1.0]),
                    label=self._generate_useless_label(),
                    phase=2
                )
            else:
                # Blue Square (PARADOX!)
                sample = CSISample(
                    color="Blue",
                    shape="Square",
                    encoding=np.array([-1.0, 1.0]),
                    label=self._generate_useless_label(),
                    phase=2
                )
            samples.append(sample)
        
        return samples
    
    def get_correlation_batch(self, batch_size: int = 32) -> List[CSISample]:
        """Get a random batch from correlation phase."""
        indices = np.random.choice(len(self.correlation_data), size=batch_size, replace=True)
        return [self.correlation_data[i] for i in indices]
    
    def get_paradox_batch(self, batch_size: int = 32) -> List[CSISample]:
        """Get a random batch from paradox phase."""
        indices = np.random.choice(len(self.paradox_data), size=batch_size, replace=True)
        return [self.paradox_data[i] for i in indices]
    
    def get_all_correlation(self) -> List[CSISample]:
        """Get all correlation phase samples."""
        return self.correlation_data
    
    def get_all_paradox(self) -> List[CSISample]:
        """Get all paradox phase samples."""
        return self.paradox_data
    
    def shuffle_correlation(self):
        """Shuffle correlation phase data."""
        np.random.shuffle(self.correlation_data)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'n_correlation': self.n_correlation,
            'n_paradox': self.n_paradox,
            'total_samples': self.n_correlation + self.n_paradox,
            'correlation_ratio': self.n_correlation / (self.n_correlation + self.n_paradox),
            'label_mode': self.label_mode,
            'unique_encodings_phase1': len(set(tuple(s.encoding) for s in self.correlation_data)),
            'unique_encodings_phase2': len(set(tuple(s.encoding) for s in self.paradox_data))
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"CSISDataset(correlation={stats['n_correlation']}, "
                f"paradox={stats['n_paradox']}, "
                f"label_mode='{stats['label_mode']}')")


def visualize_dataset(dataset: CSISDataset):
    """Visualize the CSIS dataset structure."""
    print("=" * 60)
    print("CSIS Dataset Structure")
    print("=" * 60)
    
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'Phase 1: Correlation (Rule 1)':-^60}")
    print("Red → Square: [1, 1]")
    print("Blue → Circle: [-1, -1]")
    print(f"\nSample examples:")
    for i, sample in enumerate(dataset.correlation_data[:4]):
        print(f"  {i+1}. {sample}")
    
    print(f"\n{'Phase 2: Paradox (Violates Rule 1)':-^60}")
    print("Red → Circle: [1, -1]  ← CONTRADICTS LEARNED INVARIANT!")
    print("Blue → Square: [-1, 1]  ← CONTRADICTS LEARNED INVARIANT!")
    print(f"\nSample examples:")
    for i, sample in enumerate(dataset.paradox_data[:4]):
        print(f"  {i+1}. {sample}")
    
    print("\n" + "=" * 60)
    print("External labels are USELESS (constant/random)")
    print("CMN must learn from INTERNAL consistency signals only!")
    print("=" * 60)


if __name__ == "__main__":
    # Test dataset generation
    print("Testing CSIS Dataset Generator...\n")
    
    dataset = CSISDataset(
        n_correlation=180,
        n_paradox=20,
        label_mode="constant",
        seed=42
    )
    
    visualize_dataset(dataset)
    
    print("\n\nTesting batch sampling...")
    batch = dataset.get_correlation_batch(batch_size=4)
    print(f"Correlation batch (size={len(batch)}):")
    for sample in batch:
        print(f"  {sample}")
    
    print("\nCSIS Dataset implementation complete!")
