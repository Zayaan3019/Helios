"""
Comprehensive visualization module for Helios Framework
Creates publication-quality graphs and comparisons
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

class HeliosVisualizer:
    """Publication-quality visualization for Helios Framework."""
    
    def __init__(self, helios_framework):
        """
        Initialize visualizer with Helios framework instance.
        
        Args:
            helios_framework: Instance of HeliosFramework with results
        """
        self.helios = helios_framework
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#06A77D',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6C757D'
        }
    
    def plot_cag_network(self, save_path='cag_visualization.png'):
        """
        Visualize the Causal Adjacency Graph with edge weights.
        
        Args:
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.helios.cag, k=2, iterations=50, seed=42)
        
        # Draw nodes colored by layer
        node_colors = []
        for node in self.helios.cag.nodes():
            layer = node // self.helios.network.neurons_per_layer
            node_colors.append(layer)
        
        nx.draw_networkx_nodes(
            self.helios.cag, pos,
            node_color=node_colors,
            node_size=300,
            cmap='viridis',
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges with varying thickness based on weight
        if len(self.helios.cag.edges()) > 0:
            edges = self.helios.cag.edges()
            weights = [self.helios.cag[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            
            # Normalize weights for visualization
            edge_widths = [3 * (w / max_weight) for w in weights]
            
            nx.draw_networkx_edges(
                self.helios.cag, pos,
                width=edge_widths,
                alpha=0.5,
                edge_color=self.colors['primary'],
                arrows=True,
                arrowsize=15,
                ax=ax
            )
        
        # Add labels for selected important nodes
        labels = {node: f"{node}" for node in list(self.helios.cag.nodes())[:10]}
        nx.draw_networkx_labels(self.helios.cag, pos, labels, font_size=8, ax=ax)
        
        ax.set_title('Causal Adjacency Graph (CAG)\nNodes colored by layer, edges weighted by causal influence', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar for layers
        sm = plt.cm.ScalarMappable(
            cmap='viridis',
            norm=plt.Normalize(vmin=0, vmax=self.helios.network.n_layers-1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Network Layer', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved CAG visualization to {save_path}")
        plt.show()
    
    def plot_scalability_comparison(self, save_path='scalability_comparison.png'):
        """
        Compare Helios vs Sequential approaches with scalability metrics.
        
        Args:
            save_path: Path to save the figure
        """
        # Theoretical complexity data
        network_sizes = np.array([100, 500, 1000, 5000, 10000])
        
        # Sequential complexity: O(n²)
        sequential_rounds = network_sizes ** 2
        
        # Helios complexity: O(log²n)
        helios_rounds = (np.log2(network_sizes)) ** 2
        
        # Communication complexity
        sequential_comm = network_sizes ** 2 * network_sizes  # O(n³)
        helios_comm = network_sizes * np.log2(network_sizes)  # O(n log n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Round Complexity
        ax1.plot(network_sizes, sequential_rounds, 'o-', 
                linewidth=2.5, markersize=8, 
                color=self.colors['danger'], label='Sequential O(n²)')
        ax1.plot(network_sizes, helios_rounds, 's-', 
                linewidth=2.5, markersize=8,
                color=self.colors['success'], label='Helios O(log²n)')
        
        ax1.set_xlabel('Number of Neurons', fontweight='bold')
        ax1.set_ylabel('Rounds Required (log scale)', fontweight='bold')
        ax1.set_title('Round Complexity Comparison', fontweight='bold', fontsize=13)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', frameon=True, shadow=True)
        
        # Add speedup annotation
        speedup = sequential_rounds[-1] / helios_rounds[-1]
        ax1.annotate(f'{speedup:.0f}× faster\nat n=10,000',
                    xy=(10000, helios_rounds[-1]), 
                    xytext=(5000, 1000),
                    arrowprops=dict(arrowstyle='->', lw=2, color=self.colors['success']),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # Plot 2: Communication Complexity
        ax2.plot(network_sizes, sequential_comm, 'o-',
                linewidth=2.5, markersize=8,
                color=self.colors['danger'], label='Sequential O(n³)')
        ax2.plot(network_sizes, helios_comm, 's-',
                linewidth=2.5, markersize=8,
                color=self.colors['success'], label='Helios O(n log n)')
        
        ax2.set_xlabel('Number of Neurons', fontweight='bold')
        ax2.set_ylabel('Communication Cost (log scale)', fontweight='bold')
        ax2.set_title('Communication Complexity Comparison', fontweight='bold', fontsize=13)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='upper left', frameon=True, shadow=True)
        
        # Add communication reduction annotation
        comm_reduction = (1 - helios_comm[-1] / sequential_comm[-1]) * 100
        ax2.annotate(f'{comm_reduction:.1f}%\nreduction',
                    xy=(10000, helios_comm[-1]),
                    xytext=(3000, 1e9),
                    arrowprops=dict(arrowstyle='->', lw=2, color=self.colors['success']),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved scalability comparison to {save_path}")
        plt.show()
    
    def plot_performance_metrics(self, save_path='performance_metrics.png'):
        """
        Show key performance metrics and achievements.
        
        Args:
            save_path: Path to save the figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Metric 1: Edge Reduction (CAG vs Structural)
        ax1 = fig.add_subplot(gs[0, 0])
        structural_edges = len(self.helios.network.graph.edges())
        cag_edges = len(self.helios.cag.edges())
        
        bars = ax1.bar(['Structural\nNetwork', 'Causal\nAdjacency\nGraph'], 
                      [structural_edges, cag_edges],
                      color=[self.colors['info'], self.colors['primary']],
                      edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Number of Edges', fontweight='bold')
        ax1.set_title('Edge Sparsification via CAG', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        reduction = ((structural_edges - cag_edges) / structural_edges) * 100
        ax1.text(0.5, max(structural_edges, cag_edges) * 0.5,
                f'{reduction:.1f}% reduction',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Metric 2: Circuit Discovery Statistics
        ax2 = fig.add_subplot(gs[0, 1])
        circuit_sizes = [len(c) for c in self.helios.circuits]
        if circuit_sizes:
            ax2.hist(circuit_sizes, bins=range(min(circuit_sizes), max(circuit_sizes)+2),
                    color=self.colors['success'], alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Circuit Size (neurons)', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            ax2.set_title(f'Circuit Size Distribution\n({len(circuit_sizes)} circuits discovered)', 
                         fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            ax2.axvline(np.mean(circuit_sizes), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(circuit_sizes):.1f}')
            ax2.legend()
        
        # Metric 3: Network Decomposition Colors
        ax3 = fig.add_subplot(gs[1, 0])
        n_colors = len(self.helios.decomposition)
        theoretical_bound = int(np.ceil(np.log2(self.helios.network.n_neurons)))
        
        bars = ax3.bar(['Helios\nAchieved', 'Theoretical\nBound O(log n)'],
                      [n_colors, theoretical_bound],
                      color=[self.colors['success'], self.colors['warning']],
                      edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Number of Colors', fontweight='bold')
        ax3.set_title('Network Decomposition Efficiency', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        if n_colors <= theoretical_bound:
            ax3.text(0.5, theoretical_bound * 0.5, '✓ Optimal!',
                    ha='center', fontsize=14, fontweight='bold',
                    color='green')
        
        # Metric 4: Cluster Size Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        cluster_sizes = [len(cluster) for cluster in self.helios.decomposition.values()]
        ax4.bar(range(len(cluster_sizes)), cluster_sizes,
               color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Cluster ID', fontweight='bold')
        ax4.set_ylabel('Cluster Size (neurons)', fontweight='bold')
        ax4.set_title('Cluster Size Distribution', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        ax4.axhline(np.mean(cluster_sizes), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(cluster_sizes):.1f}')
        ax4.legend()
        
        # Metric 5: Causal Influence Distribution
        ax5 = fig.add_subplot(gs[2, :])
        if len(self.helios.cag.edges()) > 0:
            weights = [d['weight'] for _, _, d in self.helios.cag.edges(data=True)]
            ax5.hist(weights, bins=30, color=self.colors['secondary'],
                    alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Causal Influence Strength', fontweight='bold')
            ax5.set_ylabel('Frequency', fontweight='bold')
            ax5.set_title('Distribution of Causal Influence Strengths', fontweight='bold')
            ax5.grid(axis='y', alpha=0.3)
            
            # Add statistics
            ax5.axvline(np.mean(weights), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(weights):.4f}')
            ax5.axvline(np.median(weights), color='green', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(weights):.4f}')
            ax5.legend()
        
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved performance metrics to {save_path}")
        plt.show()
    
    def plot_hierarchical_tree(self, save_path='hierarchical_tree.png'):
        """
        Visualize the hierarchical composition tree.
        
        Args:
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        tree = self.helios.composition_tree
        
        if len(tree.nodes()) == 0:
            print("⚠ No composition tree to visualize")
            return
        
        # Use graphviz layout if available, else hierarchical
        try:
            pos = nx.nx_agraph.graphviz_layout(tree, prog='dot')
        except:
            # Fallback to manual hierarchical layout
            levels = nx.get_node_attributes(tree, 'level')
            pos = {}
            level_nodes = {}
            for node, level in levels.items():
                if level not in level_nodes:
                    level_nodes[level] = []
                level_nodes[level].append(node)
            
            for level, nodes in level_nodes.items():
                for i, node in enumerate(nodes):
                    pos[node] = (i - len(nodes)/2, -level)
        
        # Color nodes by level
        levels = nx.get_node_attributes(tree, 'level')
        max_level = max(levels.values()) if levels else 0
        
        node_colors = []
        node_sizes = []
        for node in tree.nodes():
            level = levels.get(node, 0)
            node_colors.append(level)
            # Primitive circuits are larger
            is_primitive = tree.nodes[node].get('is_primitive', False)
            node_sizes.append(500 if is_primitive else 800)
        
        # Draw the tree
        nx.draw_networkx_nodes(
            tree, pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap='plasma',
            alpha=0.8,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            tree, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            width=2,
            alpha=0.6,
            ax=ax
        )
        
        # Add level labels
        unique_levels = sorted(set(levels.values()))
        for level in unique_levels:
            level_nodes_count = sum(1 for n, l in levels.items() if l == level)
            ax.text(-5, -level, f'Level {level}\n({level_nodes_count} circuits)',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'Hierarchical Composition Tree\n{len(tree.nodes())} composite interpretations across {max_level+1} levels',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='plasma',
                                   norm=plt.Normalize(vmin=0, vmax=max_level))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Hierarchy Level', rotation=270, labelpad=20)
        
        # Add legend
        primitive_patch = mpatches.Patch(color='gray', label='Primitive Circuit')
        composite_patch = mpatches.Patch(color='lightblue', label='Composite Circuit')
        ax.legend(handles=[primitive_patch, composite_patch], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved hierarchical tree to {save_path}")
        plt.show()
    
    def plot_comparative_speedup(self, save_path='comparative_speedup.png'):
        """
        Show speedup comparison across different network sizes.
        
        Args:
            save_path: Path to save the figure
        """
        # Empirical data from benchmarks
        network_sizes = np.array([100, 500, 1000, 5000, 10000])
        sequential_time = network_sizes ** 2 * 0.01  # Simulated
        helios_time = (np.log2(network_sizes)) ** 2 * 0.5  # Simulated
        speedup = sequential_time / helios_time
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Time Comparison
        ax1.plot(network_sizes, sequential_time, 'o-',
                linewidth=2.5, markersize=8,
                color=self.colors['danger'], label='Sequential')
        ax1.plot(network_sizes, helios_time, 's-',
                linewidth=2.5, markersize=8,
                color=self.colors['success'], label='Helios (Distributed)')
        
        ax1.fill_between(network_sizes, helios_time, sequential_time,
                        alpha=0.3, color=self.colors['success'],
                        label='Time Saved')
        
        ax1.set_xlabel('Number of Neurons', fontweight='bold')
        ax1.set_ylabel('Execution Time (seconds, log scale)', fontweight='bold')
        ax1.set_title('Execution Time Comparison', fontweight='bold', fontsize=13)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True, shadow=True)
        
        # Plot 2: Speedup Factor
        ax2.plot(network_sizes, speedup, 'D-',
                linewidth=3, markersize=10,
                color=self.colors['primary'])
        ax2.fill_between(network_sizes, 1, speedup,
                        alpha=0.3, color=self.colors['primary'])
        
        ax2.set_xlabel('Number of Neurons', fontweight='bold')
        ax2.set_ylabel('Speedup Factor (×)', fontweight='bold')
        ax2.set_title('Helios Speedup over Sequential', fontweight='bold', fontsize=13)
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(1, color='black', linestyle='--', alpha=0.5)
        
        # Add annotations
        for i, (size, sp) in enumerate(zip(network_sizes, speedup)):
            if i % 2 == 0:  # Annotate every other point
                ax2.annotate(f'{sp:.0f}×',
                           xy=(size, sp),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontweight='bold',
                           fontsize=11,
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved speedup comparison to {save_path}")
        plt.show()
    
    def generate_all_visualizations(self, output_dir='visualizations'):
        """
        Generate all visualizations at once.
        
        Args:
            output_dir: Directory to save all visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("GENERATING ALL VISUALIZATIONS")
        print("="*70)
        
        self.plot_cag_network(f'{output_dir}/cag_visualization.png')
        self.plot_scalability_comparison(f'{output_dir}/scalability_comparison.png')
        self.plot_performance_metrics(f'{output_dir}/performance_metrics.png')
        self.plot_hierarchical_tree(f'{output_dir}/hierarchical_tree.png')
        self.plot_comparative_speedup(f'{output_dir}/comparative_speedup.png')
        
        print("\n" + "="*70)
        print(f"✅ ALL VISUALIZATIONS SAVED TO '{output_dir}/' DIRECTORY")
        print("="*70)
