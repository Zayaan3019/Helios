"""
Neural Network Implementation for Helios Framework
Fixed version with proper gradient computation
"""
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List

class SyntheticNeuralNetwork:
    """Generate and manage synthetic neural networks with realistic behavior."""
    
    def __init__(self, n_neurons: int, n_layers: int, connectivity: float = 0.3):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.connectivity = connectivity
        self.neurons_per_layer = n_neurons // n_layers
        
        self.graph = self._generate_structure()
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()
        
    def _generate_structure(self) -> nx.DiGraph:
        """Generate layered DAG structure."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_neurons))
        
        # Add layer information
        for i in range(self.n_neurons):
            G.nodes[i]['layer'] = i // self.neurons_per_layer
        
        # Create feed-forward connections
        for layer in range(self.n_layers - 1):
            start_curr = layer * self.neurons_per_layer
            end_curr = (layer + 1) * self.neurons_per_layer
            start_next = end_curr
            end_next = min((layer + 2) * self.neurons_per_layer, self.n_neurons)
            
            for i in range(start_curr, end_curr):
                for j in range(start_next, end_next):
                    if np.random.rand() < self.connectivity:
                        G.add_edge(i, j)
        
        return G
    
    def _initialize_weights(self) -> Dict[Tuple[int, int], float]:
        """Initialize weights with Xavier/He initialization."""
        weights = {}
        for (i, j) in self.graph.edges():
            # Xavier initialization
            fan_in = self.graph.in_degree(j)
            fan_out = self.graph.out_degree(i)
            scale = np.sqrt(2.0 / (fan_in + fan_out + 1))
            weights[(i, j)] = np.random.randn() * scale
        return weights
    
    def _initialize_biases(self) -> Dict[int, float]:
        """Initialize biases."""
        biases = {}
        for node in range(self.n_neurons):
            if self.graph.in_degree(node) > 0:
                biases[node] = np.random.randn() * 0.01
            else:
                biases[node] = 0.0
        return biases
    
    def activate(self, inputs: np.ndarray) -> Dict[int, float]:
        """
        Forward pass through network with proper activation tracking.
        
        Args:
            inputs: Input activations for first layer neurons
            
        Returns:
            Dictionary mapping neuron ID to activation value
        """
        activations = {i: 0.0 for i in range(self.n_neurons)}
        
        # Set input layer
        for i in range(min(len(inputs), self.neurons_per_layer)):
            activations[i] = float(inputs[i])
        
        # Forward propagate through layers
        for node in nx.topological_sort(self.graph):
            if node < self.neurons_per_layer:
                continue  # Skip input layer
            
            # Compute weighted sum of inputs
            weighted_sum = self.biases.get(node, 0.0)
            for pred in self.graph.predecessors(node):
                weighted_sum += activations[pred] * self.weights[(pred, node)]
            
            # Apply activation function (tanh)
            activations[node] = np.tanh(weighted_sum)
        
        return activations
    
    def get_gradient(self, source: int, target: int, inputs: np.ndarray) -> float:
        """
        Compute gradient ∂a_target/∂a_source using backpropagation.
        
        This is the TRUE causal influence measure.
        
        Args:
            source: Source neuron ID
            target: Target neuron ID  
            inputs: Input activations
            
        Returns:
            Gradient value (causal influence strength)
        """
        # Forward pass to get all activations
        activations = self.activate(inputs)
        
        # Check if there's any path from source to target
        if not nx.has_path(self.graph, source, target):
            return 0.0
        
        # Compute gradients via reverse-mode autodiff (backprop)
        # Initialize gradients
        gradients = {i: 0.0 for i in range(self.n_neurons)}
        gradients[target] = 1.0  # ∂target/∂target = 1
        
        # Backpropagate from target to source
        for node in reversed(list(nx.topological_sort(self.graph))):
            if node == source:
                break
            
            if node < self.neurons_per_layer:
                continue  # Skip input layer
            
            # Get derivative of tanh activation
            # tanh'(x) = 1 - tanh²(x)
            activation = activations[node]
            local_grad = 1.0 - activation ** 2
            
            # Propagate gradient to predecessors
            for pred in self.graph.predecessors(node):
                weight = self.weights[(pred, node)]
                gradients[pred] += gradients[node] * local_grad * weight
        
        return abs(gradients[source])
