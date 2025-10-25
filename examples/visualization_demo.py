"""
Standalone visualization demonstration
Shows all visualization capabilities
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helios.main import HeliosFramework
from helios.visualization.plotters import HeliosVisualizer

def main():
    print("="*70)
    print("HELIOS VISUALIZATION DEMONSTRATION")
    print("="*70)
    
    # Run Helios framework
    print("\n[1/2] Running Helios Framework...")
    helios = HeliosFramework(n_neurons=200, n_layers=6, connectivity=0.3)
    helios.run_phase1(threshold=0.005, epsilon=0.05, delta=0.01)
    helios.run_phase2()
    helios.run_phase3()
    
    # Generate visualizations
    print("\n[2/2] Generating Publication-Quality Visualizations...")
    visualizer = HeliosVisualizer(helios)
    
    print("\n📊 Visualization 1: Causal Adjacency Graph")
    visualizer.plot_cag_network()
    
    print("\n📊 Visualization 2: Scalability Comparison")
    visualizer.plot_scalability_comparison()
    
    print("\n📊 Visualization 3: Performance Metrics Dashboard")
    visualizer.plot_performance_metrics()
    
    print("\n📊 Visualization 4: Hierarchical Composition Tree")
    visualizer.plot_hierarchical_tree()
    
    print("\n📊 Visualization 5: Comparative Speedup Analysis")
    visualizer.plot_comparative_speedup()
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
