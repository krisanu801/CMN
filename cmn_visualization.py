"""
CMN Visualization Suite

Comprehensive visualization tools for analyzing the CMN semantic closure experiment.

Visualizations:
1. Violation Signal Timeline (CMN Phase 1 & 2)
2. Commitment Evolution Heatmap
3. State Trajectory on Unit Circle
4. Weight Matrix Evolution
5. Permeability Gate Timeline
6. Comparative Analysis (CMN vs FNN)
7. Paradox Detection Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import seaborn as sns

from cmn_experiment import CMNExperiment


class CMNVisualizer:
    """
    Visualization suite for CMN semantic closure experiments.
    """
    
    def __init__(self, experiment: CMNExperiment):
        """
        Initialize visualizer with experiment results.
        
        Args:
            experiment: Completed CMN experiment
        """
        self.experiment = experiment
        self.results = experiment.get_results()
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_violation_timeline(self, save_path: Optional[str] = None):
        """
        Plot violation signal v(t) over time for both phases.
        
        This is the KEY plot showing semantic closure:
        - Phase 1: v → 0 (commitment forms)
        - Phase 2: v >> 0 (contradiction detected)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Phase 1: Invariant Formation
        violations_p1 = self.results['cmn_phase1']['violation']
        ax1.plot(violations_p1, linewidth=2, color='blue', alpha=0.8)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, 
                   label='Threshold (v=0.5)')
        ax1.set_xlabel('Training Step', fontsize=12)
        ax1.set_ylabel('Violation v(t)', fontsize=12)
        ax1.set_title('Phase 1: Invariant Formation (Correlation Data)', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add annotation
        final_v = violations_p1[-1]
        ax1.annotate(f'Final v = {final_v:.4f}',
                    xy=(len(violations_p1)-1, final_v),
                    xytext=(len(violations_p1)*0.7, final_v + 0.2),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=10, fontweight='bold')
        
        # Phase 2: Semantic Integrity Test
        violations_p2 = self.results['cmn_phase2']['violation']
        ax2.plot(violations_p2, linewidth=2, color='red', alpha=0.8)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
                   label='Detection Threshold')
        ax2.fill_between(range(len(violations_p2)), 0.5, violations_p2,
                        where=np.array(violations_p2) > 0.5,
                        alpha=0.3, color='red', label='Contradiction Detected')
        ax2.set_xlabel('Test Step', fontsize=12)
        ax2.set_ylabel('Violation v(t)', fontsize=12)
        ax2.set_title('Phase 2: Semantic Integrity Test (PARADOX Data)', 
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add annotation
        mean_v = np.mean(violations_p2)
        sustained = np.sum(np.array(violations_p2) > 0.5)
        ax2.annotate(f'Mean v = {mean_v:.4f}\nSustained high: {sustained}/{len(violations_p2)}',
                    xy=(len(violations_p2)*0.5, mean_v),
                    xytext=(len(violations_p2)*0.6, mean_v + 0.3),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved violation timeline to {save_path}")
        
        plt.show()
    
    def plot_commitment_evolution(self, save_path: Optional[str] = None):
        """
        Plot commitment matrix κ evolution over time.
        
        Shows how internal commitments strengthen during Phase 1.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Get commitment snapshots
        commitments_p1 = self.results['cmn_phase1']['commitment']
        
        # Initial, middle, final
        snapshots = [
            (0, 'Initial'),
            (len(commitments_p1)//2, 'Middle'),
            (len(commitments_p1)-1, 'Final')
        ]
        
        for idx, (step, label) in enumerate(snapshots):
            kappa = self.experiment.cmn.history['kappa'][step]
            
            im = axes[idx].imshow(kappa, cmap='YlOrRd', vmin=0, vmax=1)
            axes[idx].set_title(f'{label} (Step {step})\nκ_mean = {np.mean(kappa):.4f}',
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('j', fontsize=10)
            axes[idx].set_ylabel('i', fontsize=10)
            
            # Add values as text
            for i in range(kappa.shape[0]):
                for j in range(kappa.shape[1]):
                    text = axes[idx].text(j, i, f'{kappa[i, j]:.2f}',
                                        ha="center", va="center", color="black",
                                        fontsize=10)
        
        # Add colorbar
        fig.colorbar(im, ax=axes, orientation='horizontal', 
                    pad=0.1, label='Commitment κ')
        
        plt.suptitle('Commitment Matrix κ Evolution (Phase 1)', 
                    fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved commitment evolution to {save_path}")
        
        plt.show()
    
    def plot_state_trajectory(self, save_path: Optional[str] = None):
        """
        Plot state trajectory x(t) on the unit circle (for 2D case).
        
        Shows how the state evolves on the constrained manifold.
        """
        if self.experiment.cmn.n_dims != 2:
            print("State trajectory visualization only available for 2D systems")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Phase 1 trajectory
        x_history_p1 = np.array(self.experiment.cmn.history['x'][:len(self.results['cmn_phase1']['violation'])])
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=2)
        
        # Plot trajectory
        ax1.plot(x_history_p1[:, 0], x_history_p1[:, 1], 
                alpha=0.6, linewidth=1, color='blue')
        ax1.scatter(x_history_p1[0, 0], x_history_p1[0, 1], 
                   s=100, c='green', marker='o', label='Start', zorder=5)
        ax1.scatter(x_history_p1[-1, 0], x_history_p1[-1, 1], 
                   s=100, c='red', marker='s', label='End', zorder=5)
        
        ax1.set_xlabel('x₁ (Color: Red/Blue)', fontsize=12)
        ax1.set_ylabel('x₂ (Shape: Square/Circle)', fontsize=12)
        ax1.set_title('Phase 1: State Trajectory on Unit Circle', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        
        # Phase 2 trajectory
        start_idx = len(self.results['cmn_phase1']['violation'])
        x_history_p2 = np.array(self.experiment.cmn.history['x'][start_idx:start_idx+len(self.results['cmn_phase2']['violation'])])
        
        # Draw unit circle
        ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=2)
        
        # Plot trajectory
        ax2.plot(x_history_p2[:, 0], x_history_p2[:, 1], 
                alpha=0.6, linewidth=1, color='red')
        ax2.scatter(x_history_p2[0, 0], x_history_p2[0, 1], 
                   s=100, c='green', marker='o', label='Start', zorder=5)
        ax2.scatter(x_history_p2[-1, 0], x_history_p2[-1, 1], 
                   s=100, c='red', marker='s', label='End', zorder=5)
        
        ax2.set_xlabel('x₁ (Color: Red/Blue)', fontsize=12)
        ax2.set_ylabel('x₂ (Shape: Square/Circle)', fontsize=12)
        ax2.set_title('Phase 2: State Trajectory During Paradox', 
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved state trajectory to {save_path}")
        
        plt.show()
    
    def plot_comparative_analysis(self, save_path: Optional[str] = None):
        """
        Plot CMN vs FNN comparative analysis.
        
        This is the KEY comparison showing semantic closure in CMN
        vs agnosticism in FNN.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # CMN Violation (Phase 1 & 2)
        ax1 = fig.add_subplot(gs[0, 0])
        violations_p1 = self.results['cmn_phase1']['violation']
        violations_p2 = self.results['cmn_phase2']['violation']
        
        ax1.plot(violations_p1, label='Phase 1 (Correlation)', color='blue', linewidth=2)
        phase2_offset = len(violations_p1)
        ax1.plot(range(phase2_offset, phase2_offset + len(violations_p2)), 
                violations_p2, label='Phase 2 (Paradox)', color='red', linewidth=2)
        ax1.axvline(x=phase2_offset, color='black', linestyle='--', 
                   alpha=0.5, label='Phase Transition')
        ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Violation v(t)')
        ax1.set_title('CMN: Internal Violation Signal', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # FNN Loss (Phase 1 & 2)
        ax2 = fig.add_subplot(gs[0, 1])
        loss_p1 = self.results['fnn_phase1']['loss']
        loss_p2 = self.results['fnn_phase2']['loss']
        
        ax2.plot(loss_p1, label='Phase 1 (Correlation)', color='blue', linewidth=2)
        phase2_offset = len(loss_p1)
        ax2.plot(range(phase2_offset, phase2_offset + len(loss_p2)), 
                loss_p2, label='Phase 2 (Paradox)', color='red', linewidth=2)
        ax2.axvline(x=phase2_offset, color='black', linestyle='--', 
                   alpha=0.5, label='Phase Transition')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('External Loss')
        ax2.set_title('FNN: External Loss Signal (Useless)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # CMN Commitment
        ax3 = fig.add_subplot(gs[1, 0])
        commitment_p1 = self.results['cmn_phase1']['commitment']
        ax3.plot(commitment_p1, color='purple', linewidth=2)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Mean Commitment κ')
        ax3.set_title('CMN: Commitment Evolution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # FNN Weight Norm
        ax4 = fig.add_subplot(gs[1, 1])
        weight_norm_p1 = self.results['fnn_phase1']['weight_norm']
        ax4.plot(weight_norm_p1, color='orange', linewidth=2)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Total Weight Norm')
        ax4.set_title('FNN: Weight Norm (No Learning)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Phase 2 Comparison (Bar Chart)
        ax5 = fig.add_subplot(gs[2, :])
        
        cmn_mean_v = np.mean(violations_p2)
        fnn_mean_loss = np.mean(loss_p2)
        
        # Normalize for comparison
        max_val = max(cmn_mean_v, fnn_mean_loss)
        
        bars = ax5.bar(['CMN\n(Internal Violation)', 'FNN\n(External Loss)'],
                      [cmn_mean_v, fnn_mean_loss],
                      color=['red', 'blue'], alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add threshold line
        ax5.axhline(y=0.5, color='red', linestyle='--', linewidth=2,
                   label='Detection Threshold (v=0.5)')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax5.set_ylabel('Error Signal Magnitude', fontsize=12)
        ax5.set_title('Phase 2: Paradox Detection Comparison', 
                     fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation text
        if cmn_mean_v > 0.5:
            result_text = "✓ SEMANTIC CLOSURE ACHIEVED!\nCMN detects internal contradiction"
            color = 'green'
        else:
            result_text = "✗ Semantic closure not achieved"
            color = 'red'
        
        ax5.text(0.5, 0.95, result_text,
                transform=ax5.transAxes,
                fontsize=12, fontweight='bold',
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.suptitle('CMN vs FNN: Semantic Closure Comparison', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparative analysis to {save_path}")
        
        plt.show()
    
    def plot_permeability_gate(self, save_path: Optional[str] = None):
        """
        Plot permeability gate β(v) over time.
        
        Shows how external input influence is modulated by internal consistency.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Phase 1
        perm_p1 = self.results['cmn_phase1']['permeability']
        ax1.plot(perm_p1, color='green', linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Permeability β(v)')
        ax1.set_title('Phase 1: Permeability Gate (Correlation Data)', 
                     fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Phase 2
        perm_p2 = self.results['cmn_phase2']['permeability']
        ax2.plot(perm_p2, color='red', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Permeability β(v)')
        ax2.set_title('Phase 2: Permeability Gate (Paradox Data)', 
                     fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Add annotation
        mean_beta = np.mean(perm_p2)
        ax2.annotate(f'Mean β = {mean_beta:.4f}\n(Low permeability → Input rejected)',
                    xy=(len(perm_p2)*0.5, mean_beta),
                    xytext=(len(perm_p2)*0.6, mean_beta + 0.2),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved permeability gate to {save_path}")
        
        plt.show()
    
    def generate_all_plots(self, output_dir: str = "./cmn_results"):
        """
        Generate all visualization plots and save to directory.
        
        Args:
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating all plots to {output_dir}...")
        
        self.plot_violation_timeline(f"{output_dir}/violation_timeline.png")
        self.plot_commitment_evolution(f"{output_dir}/commitment_evolution.png")
        self.plot_state_trajectory(f"{output_dir}/state_trajectory.png")
        self.plot_comparative_analysis(f"{output_dir}/comparative_analysis.png")
        self.plot_permeability_gate(f"{output_dir}/permeability_gate.png")
        
        print(f"\n✓ All plots saved to {output_dir}/")


if __name__ == "__main__":
    print("Running CMN experiment and generating visualizations...\n")
    
    # Run experiment
    experiment = CMNExperiment(
        n_correlation=180,
        n_paradox=20,
        seed=42
    )
    
    results = experiment.run_full_experiment(
        phase1_steps=1000,
        phase2_steps=100,
        verbose=True
    )
    
    # Create visualizer
    visualizer = CMNVisualizer(experiment)
    
    # Generate all plots
    visualizer.generate_all_plots(output_dir="./cmn_results")
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
