"""
Phase 3: Hierarchical Composition of Interpretations
"""
import networkx as nx
from typing import List, Set, Tuple

class HierarchicalComposition:
    """Hierarchical composition protocol."""
    
    def __init__(self, cag: nx.DiGraph, circuits: List[Set[int]]):
        self.cag = cag
        self.circuits = circuits
        
    def compose(self) -> nx.DiGraph:
        """Build hierarchical composition tree."""
        print("Hierarchical composition...")
        
        composition_tree = nx.DiGraph()
        current_level = [(i, frozenset(c)) for i, c in enumerate(self.circuits)]
        
        for idx, circuit in current_level:
            composition_tree.add_node(circuit, level=0, circuit_id=idx, is_primitive=True)
        
        level = 0
        while len(current_level) > 1:
            level += 1
            next_level = []
            used = set()
            
            for i, (id1, c1) in enumerate(current_level):
                if id1 in used:
                    continue
                
                for j, (id2, c2) in enumerate(current_level[i+1:], i+1):
                    if id2 in used:
                        continue
                    
                    if self._are_composable(c1, c2):
                        merged = frozenset(c1 | c2)
                        composition_tree.add_node(merged, level=level, is_primitive=False)
                        composition_tree.add_edge(merged, c1)
                        composition_tree.add_edge(merged, c2)
                        
                        next_level.append((len(next_level), merged))
                        used.add(id1)
                        used.add(id2)
                        break
            
            for idx, circuit in current_level:
                if idx not in used:
                    next_level.append((len(next_level), circuit))
            
            print(f"  Level {level}: {len(current_level)} → {len(next_level)} circuits")
            current_level = next_level
        
        print(f"Composition tree: {level} levels")
        return composition_tree
    
    def _are_composable(self, circuit1: frozenset, circuit2: frozenset) -> bool:
        """Check if two circuits can be composed."""
        for node1 in circuit1:
            for node2 in circuit2:
                if self.cag.has_edge(node1, node2):
                    return True
        return False
