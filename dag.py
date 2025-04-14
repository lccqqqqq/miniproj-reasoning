import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Optional, Union, Tuple
from matplotlib.colors import Normalize, LinearSegmentedColormap


class DAG:
    """
    A class representing Directed Acyclic Graphs (DAGs) using lower triangular matrices.
    Supports weights, node labeling with colors, and visualization.
    """
    
    def __init__(self, n: int, weighted: bool = False):
        """
        Initialize a DAG with n nodes.
        
        Args:
            n: Number of nodes in the DAG
            weighted: Whether the DAG is weighted
        """
        self.n = n
        self.weighted = weighted
        
        # Lower triangular matrix representation
        # For unweighted DAGs: 1 means edge exists, 0 means no edge
        # For weighted DAGs: value represents edge weight, 0 means no edge
        self.adj_matrix = np.zeros((n, n), dtype=float)
        
        # Node labels and colors
        self.labels = {i: str(i) for i in range(n)}
        self.colors = {i: 'lightblue' for i in range(n)}
        
    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """
        Add a directed edge from u to v.
        
        Args:
            u: Source node
            v: Target node
            weight: Edge weight (only used if weighted=True)
        """
        if u >= v:
            raise ValueError("In a DAG, edges must go from lower to higher indices")
        
        if self.weighted:
            self.adj_matrix[u, v] = weight
        else:
            self.adj_matrix[u, v] = 1
    
    def remove_edge(self, u: int, v: int) -> None:
        """
        Remove the directed edge from u to v.
        
        Args:
            u: Source node
            v: Target node
        """
        self.adj_matrix[u, v] = 0
    
    def has_edge(self, u: int, v: int) -> bool:
        """
        Check if there is a directed edge from u to v.
        
        Args:
            u: Source node
            v: Target node
            
        Returns:
            True if edge exists, False otherwise
        """
        return self.adj_matrix[u, v] != 0
    
    def get_weight(self, u: int, v: int) -> float:
        """
        Get the weight of the edge from u to v.
        
        Args:
            u: Source node
            v: Target node
            
        Returns:
            Weight of the edge, or 0 if no edge exists
        """
        return self.adj_matrix[u, v]
    
    def set_label(self, node: int, label: str) -> None:
        """
        Set a label for a node.
        
        Args:
            node: Node index
            label: Label string
        """
        self.labels[node] = label
    
    def set_color(self, node: int, color: str) -> None:
        """
        Set a color for a node.
        
        Args:
            node: Node index
            color: Color string (can be name or hex code)
        """
        self.colors[node] = color
    
    def set_node_attributes(self, node: int, label: Optional[str] = None, color: Optional[str] = None) -> None:
        """
        Set both label and color for a node.
        
        Args:
            node: Node index
            label: Label string (optional)
            color: Color string (optional)
        """
        if label is not None:
            self.set_label(node, label)
        if color is not None:
            self.set_color(node, color)
    
    def get_children(self, node: int) -> List[int]:
        """
        Get all children of a node.
        
        Args:
            node: Node index
            
        Returns:
            List of child node indices
        """
        return [j for j in range(node + 1, self.n) if self.has_edge(node, j)]
    
    def get_parents(self, node: int) -> List[int]:
        """
        Get all parents of a node.
        
        Args:
            node: Node index
            
        Returns:
            List of parent node indices
        """
        return [i for i in range(node) if self.has_edge(i, node)]
    
    def is_acyclic(self) -> bool:
        """
        Check if the graph is acyclic.
        
        Returns:
            True if the graph is acyclic, False otherwise
        """
        # Since we enforce edges to go from lower to higher indices,
        # the graph is always acyclic
        return True
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Convert the DAG to a NetworkX graph for visualization.
        
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for i in range(self.n):
            G.add_node(i, label=self.labels[i], color=self.colors[i])
        
        # Add edges with weights if applicable
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.has_edge(i, j):
                    if self.weighted:
                        G.add_edge(i, j, weight=self.get_weight(i, j))
                    else:
                        G.add_edge(i, j)
        
        return G
    
    def visualize(self, figsize: Tuple[int, int] = (10, 8), 
                 title: str = "DAG Visualization",
                 layout: str = "hierarchical",
                 edge_colormap: str = "viridis",
                 show_colorbar: bool = True,
                 show_edge_labels: bool = False) -> None:
        """
        Visualize the DAG using matplotlib and networkx.
        
        Args:
            figsize: Figure size as (width, height)
            title: Title for the plot
            layout: Layout algorithm ('hierarchical', 'spring', 'kamada_kawai', etc.)
            edge_colormap: Colormap for edge weights
            show_colorbar: Whether to show the colorbar for edge weights
        """
        G = self.to_networkx()
        
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == "hierarchical":
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        elif layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Draw nodes
        node_colors = [G.nodes[node]["color"] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
        
        # Draw edges with colors based on weights if weighted
        if self.weighted:
            # Get edge weights
            edge_weights = [G.edges[u, v]["weight"] for u, v in G.edges()]
            
            # Create a colormap from white to red
            cmap = LinearSegmentedColormap.from_list('custom', ['white', 'red'])
            
            # Normalize weights to [0, 1] for colormap
            norm = Normalize(vmin=0, vmax=1)
            
            # Get colors for edges
            edge_colors = [cmap(norm(w)) for w in edge_weights]
            # Draw edges with colors
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                                  arrows=True, arrowsize=20, width=3)
            
            # Add edge labels
            if show_edge_labels:
                edge_labels = {(u, v): f"{G.edges[u, v]['weight']:.2f}" for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            # Add colorbar if requested
            if show_colorbar and len(edge_weights) > 0:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca())
                cbar.set_label('Edge Weight')
        else:
            # For unweighted DAGs, just draw black edges
            nx.draw_networkx_edges(G, pos, edge_color='black', 
                                  arrows=True, arrowsize=20)
        
        # Draw labels
        labels = {node: G.nodes[node]["label"] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def __str__(self) -> str:
        """String representation of the DAG."""
        return f"DAG(n={self.n}, weighted={self.weighted})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the DAG."""
        return f"DAG(n={self.n}, weighted={self.weighted}, adj_matrix=\n{self.adj_matrix})"