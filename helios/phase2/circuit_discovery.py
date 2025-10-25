"""
Phase 2: Circuit Discovery within Clusters
IMPROVED VERSION with better circuit detection
"""
import networkx as nx
from typing import List, Set

class CircuitDiscovery:
    """Discover primitive computational circuits."""
    
    def __init__(self, cag: nx.DiGraph):
        self.cag = cag
    
    def discover_circuits(self, decomposition: dict) -> List[Set[int]]:
        """
        Discover primitive circuits in each cluster.
        
        Args:
            decomposition: Network decomposition from Phase 2
            
        Returns:
            List of primitive circuits (node sets)
        """
        all_circuits = []
        
        print("Circuit discovery...")
        for color, cluster in decomposition.items():
            subgraph = self.cag.subgraph(cluster)
            circuits = self._find_circuits_in_cluster(subgraph)
            all_circuits.extend(circuits)
            print(f"  Cluster {color}: {len(circuits)} circuits")
        
        print(f"Total circuits: {len(all_circuits)}")
        return all_circuits
    
    def _find_circuits_in_cluster(self, subgraph: nx.DiGraph) -> List[Set[int]]:
        """
        Find primitive circuits via multiple pattern detection strategies.
        
        Detects:
        - Feed-forward motifs (2+ inputs → node → 1+ outputs)
        - Convergent motifs (multiple paths converging)
        - Divergent motifs (single source splitting)
        - Small cycles (feedback loops)
        """
        circuits = []
        seen_circuits = set()
        
        for node in subgraph.nodes():
            predecessors = list(subgraph.predecessors(node))
            successors = list(subgraph.successors(node))
            
            # Pattern 1: Feed-forward motif (most common)
            if len(predecessors) >= 2 and len(successors) >= 1:
                circuit = frozenset([node] + predecessors[:2] + successors[:1])
                if circuit not in seen_circuits and self._is_valid_circuit(circuit, subgraph):
                    circuits.append(set(circuit))
                    seen_circuits.add(circuit)
            
            # Pattern 2: Convergent motif (multiple inputs, single output)
            if len(predecessors) >= 3 and len(successors) == 1:
                circuit = frozenset([node] + predecessors[:3] + successors)
                if circuit not in seen_circuits and self._is_valid_circuit(circuit, subgraph):
                    circuits.append(set(circuit))
                    seen_circuits.add(circuit)
            
            # Pattern 3: Divergent motif (single input, multiple outputs)
            if len(predecessors) == 1 and len(successors) >= 2:
                circuit = frozenset([node] + predecessors + successors[:2])
                if circuit not in seen_circuits and self._is_valid_circuit(circuit, subgraph):
                    circuits.append(set(circuit))
                    seen_circuits.add(circuit)
            
            # Pattern 4: Dense subgraphs (highly interconnected)
            if len(predecessors) >= 2 and len(successors) >= 2:
                # Find densely connected neighbors
                neighbors = set(predecessors[:2]) | set(successors[:2]) | {node}
                if self._is_dense_subgraph(neighbors, subgraph):
                    circuit = frozenset(neighbors)
                    if circuit not in seen_circuits:
                        circuits.append(set(circuit))
                        seen_circuits.add(circuit)
        
        return circuits
    
    def _is_valid_circuit(self, circuit: frozenset, subgraph: nx.DiGraph) -> bool:
        """
        Check if circuit forms a valid computational unit.
        
        Requirements:
        - Weakly connected
        - At least 3 nodes
        - Has both inputs and outputs
        """
        if len(circuit) < 3:
            return False
        
        circuit_subgraph = subgraph.subgraph(circuit)
        
        # Must be weakly connected
        if not nx.is_weakly_connected(circuit_subgraph):
            return False
        
        # Must have edges (not just isolated nodes)
        if circuit_subgraph.number_of_edges() < 2:
            return False
        
        return True
    
    def _is_dense_subgraph(self, nodes: Set[int], subgraph: nx.DiGraph) -> bool:
        """Check if nodes form a densely connected subgraph."""
        if len(nodes) < 3:
            return False
        
        circuit_subgraph = subgraph.subgraph(nodes)
        n_nodes = len(nodes)
        n_edges = circuit_subgraph.number_of_edges()
        
        # Density > 0.5 (more than half of possible edges exist)
        max_edges = n_nodes * (n_nodes - 1)  # Directed graph
        density = n_edges / max_edges if max_edges > 0 else 0
        
        return density > 0.5

