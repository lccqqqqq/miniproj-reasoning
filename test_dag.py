import numpy as np
import matplotlib.pyplot as plt
from dag import DAG

def test_unweighted_dag():
    """Test creating and visualizing an unweighted DAG."""
    # Create a DAG with 5 nodes
    dag = DAG(n=5, weighted=False)
    
    # Add some edges
    dag.add_edge(0, 1)
    dag.add_edge(0, 2)
    dag.add_edge(1, 3)
    dag.add_edge(2, 3)
    dag.add_edge(2, 4)
    dag.add_edge(3, 4)
    
    # Set some node labels
    dag.set_label(0, "Start")
    dag.set_label(1, "A")
    dag.set_label(2, "B")
    dag.set_label(3, "C")
    dag.set_label(4, "End")
    
    # Set some node colors
    dag.set_color(0, "lightgreen")
    dag.set_color(4, "lightcoral")
    
    # Visualize the DAG
    dag.visualize(title="Unweighted DAG Example")
    
    # Print some information
    print(f"DAG: {dag}")
    print(f"Children of node 2: {dag.get_children(2)}")
    print(f"Parents of node 3: {dag.get_parents(3)}")
    print(f"Is acyclic: {dag.is_acyclic()}")

def test_weighted_dag():
    """Test creating and visualizing a weighted DAG."""
    # Create a weighted DAG with 6 nodes
    dag = DAG(n=6, weighted=True)
    
    # Add weighted edges
    dag.add_edge(0, 1, 2.5)
    dag.add_edge(0, 2, 1.0)
    dag.add_edge(1, 3, 3.0)
    dag.add_edge(2, 3, 1.5)
    dag.add_edge(2, 4, 2.0)
    dag.add_edge(3, 5, 4.0)
    dag.add_edge(4, 5, 1.0)
    
    # Set node labels
    dag.set_node_attributes(0, label="Source", color="lightgreen")
    dag.set_node_attributes(1, label="A", color="lightblue")
    dag.set_node_attributes(2, label="B", color="lightblue")
    dag.set_node_attributes(3, label="C", color="lightyellow")
    dag.set_node_attributes(4, label="D", color="lightyellow")
    dag.set_node_attributes(5, label="Sink", color="lightcoral")
    
    # Visualize with different layouts and colormaps
    dag.visualize(title="Weighted DAG - Hierarchical Layout", 
                 layout="hierarchical", 
                 edge_colormap="viridis")
    
    dag.visualize(title="Weighted DAG - Spring Layout", 
                 layout="spring", 
                 edge_colormap="plasma")
    
    # Try different colormaps
    dag.visualize(title="Weighted DAG - Cool Colormap", 
                 layout="hierarchical", 
                 edge_colormap="cool")
    
    dag.visualize(title="Weighted DAG - Hot Colormap", 
                 layout="hierarchical", 
                 edge_colormap="hot")
    
    # Print some information
    print(f"Weighted DAG: {dag}")
    print(f"Edge weight from 0 to 1: {dag.get_weight(0, 1)}")
    print(f"Edge weight from 2 to 3: {dag.get_weight(2, 3)}")

def test_dag_operations():
    """Test various DAG operations."""
    # Create a DAG
    dag = DAG(n=4, weighted=True)
    
    # Add edges
    dag.add_edge(0, 1, 1.0)
    dag.add_edge(0, 2, 2.0)
    dag.add_edge(1, 3, 3.0)
    dag.add_edge(2, 3, 4.0)
    
    # Check edge existence
    print(f"Edge 0->1 exists: {dag.has_edge(0, 1)}")
    print(f"Edge 1->0 exists: {dag.has_edge(1, 0)}")  # Should be False
    
    # Remove an edge
    dag.remove_edge(0, 1)
    print(f"After removal, edge 0->1 exists: {dag.has_edge(0, 1)}")
    
    # Add it back
    dag.add_edge(0, 1, 5.0)
    print(f"After adding back, edge 0->1 weight: {dag.get_weight(0, 1)}")
    
    # Try to add an invalid edge (should raise ValueError)
    try:
        dag.add_edge(2, 1)  # This should fail
    except ValueError as e:
        print(f"Expected error: {e}")

if __name__ == "__main__":
    print("Testing unweighted DAG:")
    test_unweighted_dag()
    
    print("\nTesting weighted DAG:")
    test_weighted_dag()
    
    print("\nTesting DAG operations:")
    test_dag_operations() 