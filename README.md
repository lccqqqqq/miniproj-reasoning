# DAG Implementation

This project provides a Python implementation of Directed Acyclic Graphs (DAGs) based on lower triangular matrices. The implementation supports weighted edges, node labeling with colors, and visualization capabilities.

## Features

- **Lower Triangular Matrix Representation**: Efficiently represents DAGs using lower triangular matrices
- **Weighted Edges**: Support for both weighted and unweighted DAGs
- **Node Labeling**: Custom labels for nodes
- **Node Coloring**: Custom colors for nodes
- **Visualization**: Multiple layout options for visualizing DAGs
- **Graph Operations**: Add/remove edges, check edge existence, get children/parents

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Note: For hierarchical layout visualization, you need to have Graphviz installed on your system:
- On Ubuntu/Debian: `sudo apt-get install graphviz graphviz-dev`
- On macOS: `brew install graphviz`
- On Windows: Download from [Graphviz website](https://graphviz.org/download/)

## Usage

### Basic Usage

```python
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
```

### Weighted DAG

```python
# Create a weighted DAG
dag = DAG(n=4, weighted=True)

# Add weighted edges
dag.add_edge(0, 1, 2.5)
dag.add_edge(0, 2, 1.0)
dag.add_edge(1, 3, 3.0)
dag.add_edge(2, 3, 1.5)

# Visualize with different layouts
dag.visualize(title="Weighted DAG", layout="hierarchical")
dag.visualize(title="Weighted DAG", layout="spring")
```

### Available Layouts

- `hierarchical`: Hierarchical layout (requires Graphviz)
- `spring`: Force-directed layout
- `kamada_kawai`: Kamada-Kawai layout

## Running Tests

To run the example tests:

```bash
python test_dag.py
```

## Implementation Details

The DAG is implemented using a lower triangular matrix, which ensures that the graph is always acyclic. Edges can only go from lower indices to higher indices, preventing cycles.

For unweighted DAGs, the matrix contains 1s for edges and 0s for non-edges. For weighted DAGs, the matrix contains the edge weights.

## License

MIT 