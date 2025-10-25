"""
Phase 2: Network Decomposition using Miller's Algorithm
"""
import numpy as np
import networkx as nx
from typing import Dict, Set

class NetworkDecomposition:
    """Distributed network decomposition."""
    
    def __init__(self, cag: nx.DiGraph, diameter_bound: int = None):
        self.cag = cag
        self.n = len(cag.nodes())
        self.diameter_bound = diameter_bound or int(np.ceil(np.log2(self.n)))
        
    def decompose(self) -> Dict[int, Set[int]]:
        """Compute network decomposition."""
        print(f"Network decomposition (diameter ≤ {self.diameter_bound})...")
        
        uncolored = set(self.cag.nodes())
        decomposition = {}
        color = 0
        
        while uncolored:
            leaders = self._elect_leaders(uncolored)
            clusters = self._form_clusters(leaders, uncolored)
            
            decomposition[color] = clusters
            uncolored -= clusters
            
            print(f"  Color {color}: {len(clusters)} nodes, {len(uncolored)} remaining")
            color += 1
        
        print(f"Decomposition: {color} colors (bound: O(log n) = {int(np.ceil(np.log2(self.n)))})")
        return decomposition
    
    def _elect_leaders(self, candidates: Set[int]) -> Set[int]:
        """Randomized leader election."""
        leaders = set()
        priorities = {v: np.random.rand() for v in candidates}
        
        for v in candidates:
            neighbors = set(self.cag.neighbors(v)) & candidates
            if all(priorities[v] > priorities.get(u, 0) for u in neighbors):
                leaders.add(v)
        
        return leaders
    
    def _form_clusters(self, leaders: Set[int], candidates: Set[int]) -> Set[int]:
        """Form clusters around leaders."""
        clusters = set()
        
        for leader in leaders:
            visited = {leader}
            queue = [(leader, 0)]
            
            while queue:
                node, dist = queue.pop(0)
                if dist >= self.diameter_bound // 2:
                    continue
                
                for neighbor in self.cag.neighbors(node):
                    if neighbor in candidates and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            clusters.update(visited)
        
        return clusters
