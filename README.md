# Constraint-Manifold Networks (CMN)

**Semantic Closure in Constraint-Manifold Networks: A Foundational and Hierarchical Analysis**

## Overview

Standard neural networks operate as statistical correlation machines, optimizing external loss functions without regard for internal structural integrity. This dependence on external supervision prevents **Semantic Closure**: the capacity of a system to autonomously detect and resolve contradictions between an input and its self-imposed normative constraints.

This repository contains the official implementation of the **Constraint-Manifold Network (CMN)** and its hierarchical extension (**H-CMN**).

The CMN is a dynamical system architecture that replaces function approximation with invariant preservation on a restricted hypersphere manifold. It introduces novel mechanisms for:

* **Intrinsic Normativity:** Detecting "impossible" inputs without external labels.
* **Contextual Governance:** Autonomously resolving paradoxes via top-down constraint modulation.
![Architectural Evolution](main.png)
## Key Concepts

### 1. The Single-Layer CMN

* **Goal:** Optimize internal integrity (Invariance) rather than external accuracy (Correlation).
* **Commitment Metric (\kappa):** A measure of internal belief rigidity. As the network learns, it transitions from plastic to rigid, locking in structural invariants.
* **Violation Signal (v):** An internal error signal derived from the system's failure to maintain equilibrium on its own manifold.
* **Semantic Closure:** The ability to detect anomalies (e.g., a "Red Circle" when the rule is "Red Square") purely through internal stress (v \gg 0), even when external labels are useless.

### 2. The Hierarchical CMN (H-CMN)

* **Contextual Governance:** A higher executive layer (L_2) monitors the distress of the sensory layer (L_1).
* **Empathy Gradient :** A mathematical primitive that allows L_2 to "feel" L_1's violation and adjust the context state x_2 to resolve it.
* **Phase Transition:** In response to a rule change (Context Switch), L_2 executes a rapid state shift to restructure L_1's constraints, solving the paradox dynamically.

## Repository Structure

### Core Network Implementations

* `cmn_network.py`: Implementation of the Single-Layer CMN with multi-timescale dynamics (Fast state evolution, Intermediate weight learning, Slow commitment updates).
* `hcmn_network.py`: Implementation of the Hierarchical CMN (H-CMN) featuring the Empathy Gradient and dynamic weight tensors.
* `fnn_baseline.py`: A standard Feedforward Neural Network baseline for comparison.

### Experiments

* `cmn_experiment.py`: Runs the **Semantic Closure Experiment**. Tests if the network can detect contradictions in a "Correlation-Paradox" task.
* `hcmn_experiment.py`: Runs the **Context Switching Experiment**. Tests if the H-CMN can adapt to abrupt rule changes without warning.

### Datasets & Utilities

* `csis_dataset.py`: Generates the Color-Shape Invariant Set (CSIS) for semantic closure testing.
* `context_switch_dataset.py`: Generates a continuous stream of data with abrupt rule reversals.
* `cmn_utils.py`: Utility functions for metric tracking and testing.

### Visualization

* `cmn_visualization.py`: Generates plots for the Single-Layer experiments (Violation Timeline, Comparative Analysis).
* `hcmn_visualization.py`: Generates plots for the H-CMN experiments (Phase Transitions, Empathy Coupling).

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/krisanu801/CMN.git
cd CMN
pip install numpy matplotlib seaborn

```

## Usage

### 1. Reproduce Semantic Closure (Single-Layer CMN)

Compare the CMN against a standard FNN baseline on a task where external labels are rendered useless.

```bash
python cmn_experiment.py

```

* **Phase 1 (Training):** CMN forms an invariant (\kappa \to 0.6).
* **Phase 2 (Testing):** CMN detects the paradox (v \approx 0.86). FNN fails silently.

To visualize the results:

```bash
python cmn_visualization.py

```

*Outputs plots to `./cmn_results/*`

### 2. Reproduce Contextual Governance (H-CMN)

Test the Hierarchical system on a Context Switching task where rules flip at step 500.

```bash
python hcmn_experiment.py

```

* **Switch Event:** L_1 reports a violation spike.
* **Resolution:** L_2 executes a phase transition driven by the Empathy Gradient.

To visualize the results:

```bash
python hcmn_visualization.py

```

*Outputs plots to `./hcmn_results/*`

## Results Summary

| Metric | Standard FNN | CMN / H-CMN |
| --- | --- | --- |
| **Paradox Detection** | **Failed** (Silent) | **Success** (High v spike) |
| **Error Source** | External Label | Internal Constraint |
| **Response to Change** | Catastrophic Forgetting | **Contextual Phase Transition** |

## Citation

If you use this code in your research, please cite the paper:

```bibtex
@article{sarkar2025cmn,
  title={Semantic Closure in Constraint-Manifold Networks: A Foundational and Hierarchical Analysis},
  author={Sarkar, Krisanu},
  institution={Indian Institute of Technology Bombay},
  year={2025}
}

```

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

Copyright (c) 2025 krisanu801
