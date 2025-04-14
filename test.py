#%%
from dag import DAG


# Create an unweighted DAG with 5 nodes
dag = DAG(n=5, weighted=False)

# Add edges
dag.add_edge(0, 1)
dag.add_edge(0, 2)
dag.add_edge(1, 3)
dag.add_edge(2, 3)
dag.add_edge(2, 4)
dag.add_edge(3, 4)

# Set node labels
dag.set_label(0, "Start")
dag.set_label(4, "End")

# Set node colors
dag.set_color(0, "lightgreen")
dag.set_color(4, "lightcoral")

# Visualize the DAG
dag.visualize(title="My DAG")

dag = DAG(n=4, weighted=True)

# Add weighted edges
dag.add_edge(0, 1, 1.0)
dag.add_edge(0, 2, 0.2)
dag.add_edge(1, 3, 0.5)
dag.add_edge(2, 3, 0.5)

# Visualize with a specific colormap
dag.visualize(title="Weighted DAG", 
             layout="hierarchical", 
             edge_colormap="RdGy")