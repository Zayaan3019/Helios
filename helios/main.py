"""
Main Helios Framework Class
"""
import numpy as np
from helios.core.neural_network import SyntheticNeuralNetwork
from helios.phase1.cag_construction import DecentralizedCAGConstruction
from helios.phase2.network_decomposition import NetworkDecomposition
from helios.phase2.circuit_discovery import CircuitDiscovery
from helios.phase3.hierarchical_composition import HierarchicalComposition

class HeliosFramework:
    """Complete Helios framework pipeline."""
    
    def __init__(self, n_neurons: int = 100, n_layers: int = 5, connectivity: float = 0.3):
        print("="*60)
        print("HELIOS FRAMEWORK FOR DISTRIBUTED MECHANISTIC INTERPRETABILITY")
        print("="*60)
        
        print(f"\n[Step 0] Generating synthetic network...")
        self.network = SyntheticNeuralNetwork(n_neurons, n_layers, connectivity)
        print(f"  {n_neurons} neurons, {n_layers} layers, {len(self.network.graph.edges())} edges")
        
    def run_phase1(self, threshold=0.01, epsilon=0.05, delta=0.01):
        """Execute Phase 1: CAG Construction."""
        print("\n" + "="*60)
        print("[PHASE 1] CAG CONSTRUCTION")
        print("="*60)
        
        phase1 = DecentralizedCAGConstruction(self.network, threshold, epsilon, delta)
        self.cag = phase1.construct_cag()
        
        print(f"\nPhase 1 Results:")
        print(f"  CAG edges: {len(self.cag.edges())}")
        print(f"  Sparsity: {len(self.cag.edges())/len(self.network.graph.edges()):.2%}")
        return self.cag
    
    def run_phase2(self, diameter_bound=None):
        """Execute Phase 2."""
        print("\n" + "="*60)
        print("[PHASE 2] NETWORK DECOMPOSITION & CIRCUIT DISCOVERY")
        print("="*60)
        
        decomposer = NetworkDecomposition(self.cag, diameter_bound)
        self.decomposition = decomposer.decompose()
        
        discoverer = CircuitDiscovery(self.cag)
        self.circuits = discoverer.discover_circuits(self.decomposition)
        
        print(f"\nPhase 2 Results:")
        print(f"  Colors: {len(self.decomposition)}")
        print(f"  Circuits: {len(self.circuits)}")
        return self.decomposition, self.circuits
    
    def run_phase3(self):
        """Execute Phase 3."""
        print("\n" + "="*60)
        print("[PHASE 3] HIERARCHICAL COMPOSITION")
        print("="*60)
        
        composer = HierarchicalComposition(self.cag, self.circuits)
        self.composition_tree = composer.compose()
        return self.composition_tree
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "="*60)
        print("HELIOS ANALYSIS REPORT")
        print("="*60)
        
        n = self.network.n_neurons
        print(f"\n[Network Statistics]")
        print(f"  Neurons: {n}")
        print(f"  Structural edges: {len(self.network.graph.edges())}")
        
        print(f"\n[Theoretical Bounds]")
        print(f"  Round complexity: O(log²n) = {int(np.ceil(np.log2(n)**2))}")
        print(f"  Communication: O(n log n) = {int(n * np.ceil(np.log2(n)))}")
        
        print(f"\n[Achieved Results]")
        print(f"  CAG edges: {len(self.cag.edges())}")
        print(f"  Decomposition colors: {len(self.decomposition)}")
        print(f"  Primitive circuits: {len(self.circuits)}")
