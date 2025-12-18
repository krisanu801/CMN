"""
H-CMN Visualization Suite

Generates comprehensive visualizations for the Hierarchical CMN context switching experiment.
Shows violation spikes, phase transitions, empathy coupling, and contextual governance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os

from hcmn_experiment import HCMNExperiment


class HCMNVisualizer:
    """
    Visualization suite for H-CMN experimental results.
    """
    
    def __init__(self, experiment: HCMNExperiment, output_dir: str = "hcmn_results"):
        """
        Initialize visualizer.
        
        Args:
            experiment: Completed HCMNExperiment instance
            output_dir: Directory to save plots
        """
        self.experiment = experiment
        self.results = experiment.get_results()
        self.switch_point = experiment.switch_point
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_violation_timeline(self, save: bool = True):
        """
        Plot v₁ and v₂ over time showing the context switch spike and decay.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        steps = self.results['step']
        v1 = self.results['v1']
        v2 = self.results['v2']
        
        # Plot v₁ (L1 violation)
        ax1.plot(steps, v1, 'b-', linewidth=2, label='v₁ (L1 violation)', alpha=0.8)
        ax1.axvline(self.switch_point, color='r', linestyle='--', linewidth=2, 
                   label='Context Switch', alpha=0.7)
        ax1.axhline(0.5, color='orange', linestyle=':', linewidth=1.5, 
                   label='Spike Threshold (0.5)', alpha=0.6)
        ax1.fill_between(steps, 0, v1, alpha=0.2, color='blue')
        ax1.set_ylabel('L1 Violation (v₁)', fontsize=12, fontweight='bold')
        ax1.set_title('H-CMN Context Switch Detection: Violation Timeline', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Annotate spike
        spike_idx = self.switch_point
        if spike_idx < len(v1):
            ax1.annotate(f'Spike: v₁={v1[spike_idx]:.2f}',
                        xy=(spike_idx, v1[spike_idx]),
                        xytext=(spike_idx + 100, v1[spike_idx] + 0.3),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=11, fontweight='bold', color='red')
        
        # Plot v₂ (L2 coupled violation)
        ax2.plot(steps, v2, 'g-', linewidth=2, label='v₂ (L2 coupled violation)', alpha=0.8)
        ax2.axvline(self.switch_point, color='r', linestyle='--', linewidth=2, 
                   label='Context Switch', alpha=0.7)
        ax2.fill_between(steps, 0, v2, alpha=0.2, color='green')
        ax2.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax2.set_ylabel('L2 Violation (v₂)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'violation_timeline.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
        return fig
    
    def plot_phase_transition(self, save: bool = True):
        """
        Plot L2 state changes (x₂) showing the phase transition at context switch.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        steps = self.results['step']
        x2_change = self.results['x2_norm_change']
        
        # Plot x₂ change magnitude
        ax.plot(steps, x2_change, 'purple', linewidth=2, label='||Δx₂|| (L2 state change)', alpha=0.8)
        ax.axvline(self.switch_point, color='r', linestyle='--', linewidth=2, 
                  label='Context Switch', alpha=0.7)
        ax.axhline(0.3, color='orange', linestyle=':', linewidth=1.5, 
                  label='Phase Transition Threshold (0.3)', alpha=0.6)
        ax.fill_between(steps, 0, x2_change, alpha=0.2, color='purple')
        
        # Highlight adaptation window
        ax.axvspan(self.switch_point, self.switch_point + 100, 
                  alpha=0.1, color='yellow', label='Adaptation Window')
        
        ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('L2 State Change (||Δx₂||)', fontsize=12, fontweight='bold')
        ax.set_title('H-CMN Phase Transition: L2 Context Modulation', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotate max transition
        max_idx = np.argmax(x2_change)
        max_val = x2_change[max_idx]
        ax.annotate(f'Max Transition: {max_val:.2f}',
                   xy=(steps[max_idx], max_val),
                   xytext=(steps[max_idx] + 100, max_val + 0.2),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                   fontsize=11, fontweight='bold', color='purple')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'phase_transition.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
        return fig
    
    def plot_empathy_coupling(self, save: bool = True):
        """
        Plot empathy gradient and correlation between v₁ and v₂.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        steps = self.results['step']
        empathy_grad = self.results['empathy_gradient_norm']
        v1 = np.array(self.results['v1'])
        v2 = np.array(self.results['v2'])
        v2_local = np.array(self.results['v2_local'])
        
        # Plot empathy gradient
        ax1.plot(steps, empathy_grad, 'orange', linewidth=2, 
                label='||∇_{x₂}v₁|| (Empathy Gradient)', alpha=0.8)
        ax1.axvline(self.switch_point, color='r', linestyle='--', linewidth=2, 
                   label='Context Switch', alpha=0.7)
        ax1.fill_between(steps, 0, empathy_grad, alpha=0.2, color='orange')
        ax1.set_ylabel('Empathy Gradient Norm', fontsize=12, fontweight='bold')
        ax1.set_title('H-CMN Empathy Mechanism: L2 Response to L1 Struggles', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot v₁ vs empathy component
        # Empathy component = v₂ - v₂_local = λv₁
        empathy_component = v2 - v2_local
        lambda_v1 = self.experiment.hcmn.lambda_coupling * v1
        
        ax2.plot(steps, lambda_v1, 'b-', linewidth=2, label='λv₁ (Expected Empathy)', alpha=0.7)
        ax2.plot(steps, empathy_component, 'g--', linewidth=2, 
                label='v₂ - v₂_local (Actual Empathy)', alpha=0.7)
        ax2.axvline(self.switch_point, color='r', linestyle='--', linewidth=2, 
                   label='Context Switch', alpha=0.7)
        ax2.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Empathy Component', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add correlation text
        adaptation_window = slice(self.switch_point, min(self.switch_point + 100, len(v1)))
        corr = np.corrcoef(lambda_v1[adaptation_window], empathy_component[adaptation_window])[0, 1]
        ax2.text(0.02, 0.98, f'Adaptation Window Correlation: {corr:.3f}',
                transform=ax2.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'empathy_coupling.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
        return fig
    
    def plot_comprehensive_timeline(self, save: bool = True):
        """
        Comprehensive multi-panel plot showing all key metrics.
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        
        steps = self.results['step']
        v1 = self.results['v1']
        v2 = self.results['v2']
        x2_change = self.results['x2_norm_change']
        empathy_grad = self.results['empathy_gradient_norm']
        
        # Panel 1: Violations
        axes[0].plot(steps, v1, 'b-', linewidth=2, label='v₁ (L1)', alpha=0.8)
        axes[0].plot(steps, v2, 'g-', linewidth=2, label='v₂ (L2)', alpha=0.8)
        axes[0].axvline(self.switch_point, color='r', linestyle='--', linewidth=2, alpha=0.7)
        axes[0].axhline(0.5, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
        axes[0].set_ylabel('Violation', fontsize=11, fontweight='bold')
        axes[0].set_title('H-CMN Contextual Governance: Complete Timeline', 
                         fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Phase Transition
        axes[1].plot(steps, x2_change, 'purple', linewidth=2, label='||Δx₂||', alpha=0.8)
        axes[1].axvline(self.switch_point, color='r', linestyle='--', linewidth=2, alpha=0.7)
        axes[1].axhline(0.3, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
        axes[1].set_ylabel('L2 State Change', fontsize=11, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        # Panel 3: Empathy Gradient
        axes[2].plot(steps, empathy_grad, 'orange', linewidth=2, label='||∇_{x₂}v₁||', alpha=0.8)
        axes[2].axvline(self.switch_point, color='r', linestyle='--', linewidth=2, alpha=0.7)
        axes[2].set_ylabel('Empathy Gradient', fontsize=11, fontweight='bold')
        axes[2].legend(loc='upper right', fontsize=9)
        axes[2].grid(True, alpha=0.3)
        
        # Panel 4: Rule Labels
        rules = self.results['rule']
        rule_numeric = [0 if r == "Rule A" else 1 for r in rules]
        axes[3].fill_between(steps, 0, rule_numeric, alpha=0.3, 
                            color=['blue' if r == 0 else 'red' for r in rule_numeric],
                            step='mid')
        axes[3].axvline(self.switch_point, color='r', linestyle='--', linewidth=2, 
                       label='Context Switch', alpha=0.7)
        axes[3].set_yticks([0, 1])
        axes[3].set_yticklabels(['Rule A\n(Red→Square)', 'Rule B\n(Red→Circle)'])
        axes[3].set_xlabel('Time Step', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Active Rule', fontsize=11, fontweight='bold')
        axes[3].legend(loc='upper right', fontsize=9)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'comprehensive_timeline.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
        return fig
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("\n" + "="*70)
        print("GENERATING H-CMN VISUALIZATIONS")
        print("="*70)
        
        print("\n1. Violation Timeline...")
        self.plot_violation_timeline()
        
        print("\n2. Phase Transition...")
        self.plot_phase_transition()
        
        print("\n3. Empathy Coupling...")
        self.plot_empathy_coupling()
        
        print("\n4. Comprehensive Timeline...")
        self.plot_comprehensive_timeline()
        
        print("\n" + "="*70)
        print(f"All plots saved to: {self.output_dir}/")
        print("="*70)


if __name__ == "__main__":
    print("Running H-CMN experiment and generating visualizations...\n")
    
    # Run experiment
    experiment = HCMNExperiment(n_steps=1000, switch_point=500, seed=42)
    experiment.run_full_experiment(verbose=True)
    
    # Generate visualizations
    visualizer = HCMNVisualizer(experiment)
    visualizer.generate_all_plots()
    
    print("\n✓ Visualization complete!")
