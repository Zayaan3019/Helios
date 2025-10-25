"""
Enhanced basic usage example with comprehensive visualizations
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helios.main import HeliosFramework
from helios.visualization.plotters import HeliosVisualizer

def main():
    # Initialize framework
    print("Initializing Helios Framework...")
    print("="*70)
    helios = HeliosFramework(n_neurons=150, n_layers=6, connectivity=0.35)
    
    # Run all three phases
    print("\n" + "="*70)
    print("PHASE 1: Causal Adjacency Graph Construction")
    print("="*70)
    cag = helios.run_phase1(threshold=0.005, epsilon=0.05, delta=0.01)
    
    print("\n" + "="*70)
    print("PHASE 2: Network Decomposition & Circuit Discovery")
    print("="*70)
    decomposition, circuits = helios.run_phase2()
    
    print("\n" + "="*70)
    print("PHASE 3: Hierarchical Composition")
    print("="*70)
    composition_tree = helios.run_phase3()
    
    # Generate text report
    print("\n" + "="*70)
    print("FINAL ANALYSIS REPORT")
    print("="*70)
    helios.generate_report()
    
    # === NEW: GENERATE VISUALIZATIONS ===
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    visualizer = HeliosVisualizer(helios)
    visualizer.generate_all_visualizations(output_dir='results/visualizations')
    
    print("\n" + "="*70)
    print("✅ HELIOS EXECUTION COMPLETE WITH VISUALIZATIONS")
    print("="*70)
    print("\nGenerated Files:")
    print("  📊 results/visualizations/cag_visualization.png")
    print("  📊 results/visualizations/scalability_comparison.png")
    print("  📊 results/visualizations/performance_metrics.png")
    print("  📊 results/visualizations/hierarchical_tree.png")
    print("  📊 results/visualizations/comparative_speedup.png")
    print("="*70)

if __name__ == "__main__":
    main()


