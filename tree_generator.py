#%%
import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout 
def create_random_rooted_tree(n, vertex_concepts=None, edge_concepts=None):
    """
    Create a random rooted tree with n vertices.
    
    Parameters:
    -----------
    n : int
        Number of vertices in the tree
    vertex_concepts : list, optional
        List of concepts to associate with vertices
    edge_concepts : list, optional
        List of concepts to associate with edges
    
    Returns:
    --------
    G : nx.DiGraph
        A directed graph representing the rooted tree
    """
    # Create an empty directed graph
    G = nx.DiGraph()
    
    # Add the root node (0)
    G.add_node(0)
    
    # Generate the random tree structure
    for i in range(1, n):
        # Choose a random existing node as parent
        parent = random.choice(range(i))
        G.add_edge(parent, i)
    
    # Associate vertices with concepts if provided
    if vertex_concepts:
        # If fewer concepts than vertices, cycle through the concepts
        vertex_data = {i: vertex_concepts[i % len(vertex_concepts)] for i in range(n)}
        nx.set_node_attributes(G, vertex_data, 'concept')
    
    # Associate edges with concepts if provided
    if edge_concepts:
        edge_data = {}
        edges = list(G.edges())
        for i, (u, v) in enumerate(edges):
            edge_data[(u, v)] = edge_concepts[i % len(edge_concepts)]
        nx.set_edge_attributes(G, edge_data, 'concept')
    
    return G

def visualize_tree(G):
    """
    Visualize the tree with node and edge labels.
    """
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes
    
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
    
    # Draw node labels
    node_labels = {}
    for node in G.nodes():
        label = f"{node}"
        if 'concept' in G.nodes[node]:
            label += f"\n({G.nodes[node]['concept']})"
        node_labels[node] = label
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    
    # Draw edge labels
    edge_labels = {}
    for u, v in G.edges():
        if 'concept' in G[u][v]:
            edge_labels[(u, v)] = G[u][v]['concept']
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#%%

# Visualize the tree
# visualize_tree(tree)

pos = graphviz_layout(tree, prog='dot', root=0)
nx.draw(tree, pos, with_labels=True)
plt.show()

#%% Constructing ficticious concepts
tree = create_random_rooted_tree(15)
pos = graphviz_layout(tree, prog='dot', root=0)
nx.draw(tree, pos, with_labels=True)
plt.show()

#%% a list of ficticious entities and their attributes

ficticious_entities = [
    ["wumpus", "yumpus", "zumpus", "dumpus", "rompus", "numpus", "tumpus", "vumpus", "fimpus", "jompus"],
    ["timple", "yimple", "starple", "shumple", "zhomple", "remple", "fomple", "fimple", "worple", "sorple"],
    ["tergit", "gergit", "stergit", "kergit", "shergit", "pergit", "bongit", "orgit", "welgit", "jelgit"],
    ["felper", "dolper", "sarper", "qirper", "chorper", "parper", "karper", "lemper", "hilper", "gomper"],
    ["dalpist", "lumpist", "rifpist", "storpist", "shalpist", "yerpist", "zilpist", "boompist", "scrompist", "phorpist"],
    ["prilpant", "gwompant", "nurpant", "grimpant", "shilpant", "zhorpant", "rorpant", "dropant", "lerpant", "quimpant"],
    ["zilpor", "frompor", "stirpor", "porpor", "kurpor", "shampor", "werpor", "zhimpor", "yempor", "jempor"],
    ["folpee", "drompee", "delpee", "lompee", "wolpee", "gorpee", "shimpee", "rimpee", "twimpee", "serpee"],
    ["daumpin", "thorpin", "borpin", "rofpin", "bempin", "dulpin", "harpin", "lirpin", "yompin", "stopin"]
]

ficticious_entities = [item for sublist in ficticious_entities for item in sublist]

attibutes = [
    ["blue", "red", "brown", "orange"],
    ["small", "large"],
    ["metallic", "wooden", "luminous", "liquid"],
    ["transparent", "opaque"],
    ["nervous", "happy", "feisty", "shy"],
    ["bright", "dull"],
    ["sweet", "sour", "spicy", "bitter"],
    ["floral", "fruity", "earthy"],
    ["hot", "cold", "temperate"],
    ["kind", "mean", "angry", "amenable", "aggressive"],
    ["melodic", "muffled", "discordant", "loud"],
    ["slow", "moderate", "fast"],
    ["windy", "sunny", "overcast", "rainy", "snowy"]
]

attributes = ["scorching", "crimson", "melodic", "fragrant", "velvety", "bitter", "towering", "slick", "jovial", "ancient", "chilly", "emerald", "rhythmic", "pungent", "silky", "sour", "massive", "greasy", "cheerful", "medieval", "sweltering", "amber", "harmonious", "musty", "coarse", "salty", "colossal", "slippery", "mirthful", "timeless"]

attibutes = [item for sublist in attibutes for item in sublist]

random_person_name = ["Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren", "John", "Jane", "Jim", "Jack", "Jill", "Chuqiao"]

#%% Assigning entities to nodes
import random
import networkx as nx

vertex_entities = random.sample(ficticious_entities, len(tree.nodes()))
vertex_attributes = random.sample(attibutes, len(tree.nodes()))

vertex_concepts = [(entity, attribute) for entity, attribute in zip(vertex_entities, vertex_attributes)]

nx.set_node_attributes(tree, {i: concept for i, concept in enumerate(vertex_concepts)}, 'concept')
pos = graphviz_layout(tree, prog='dot', root=0)
nx.draw(tree, pos, with_labels=True)
plt.show()

#%%
# Get concepts from tree nodes
concepts = nx.get_node_attributes(tree, 'concept')

# Print concepts for each node
for node_id, concept in concepts.items():
    print(f"Node {node_id}: Entity '{concept[0]}' with attribute '{concept[1]}'")


# %% Construct prompts out of the tree

prefix = ["Every", "Any", "Each"]
plural_form = lambda x : x + "s" if x[-1] != "s" else x + "es"

edge_description = lambda edge : f"{random.choice(prefix)} {concepts[edge[0]][0]} is a {concepts[edge[1]][0]}. "

attribute_description = lambda node : f"All {plural_form(concepts[node][0])} are {'not ' if random.random() < 0.5 else ''}{concepts[node][1]}. "

description = [
    f"{edge_description(edge)}" for edge in tree.edges()
] + [
    f"{attribute_description(node)}" for node in tree.nodes()
]
description = " ".join(description)

#%% Formulating the question

# need to choose upstream entities and ask for downstream attributes
def get_ancestors(tree, node):
    """Get all ancestors of a node in the tree (nodes on path from root to node)"""
    path = []
    current = node
    while current in tree.nodes():
        path.append(current)
        # Get predecessors (parents) - should only be 1 in a tree
        predecessors = list(tree.predecessors(current))
        if not predecessors:
            break
        current = predecessors[0]
    return path[::-1]  # Reverse to get root->node order

def get_descendants(tree, node):
    """Get all descendants of a node in the tree (nodes in subtree rooted at node)"""
    return list(nx.descendants(tree, node))

def get_node_distance(tree, node1, node2):
    """Get the distance between two nodes in the tree (length of path between them)"""
    try:
        return nx.shortest_path_length(tree, node1, node2)
    except nx.NetworkXNoPath:
        return float('inf')

#%% Start from the root node and end at one of the leaves

def get_leaf_nodes(tree):
    """Get all leaf nodes (nodes with no successors) in the tree"""
    return [node for node in tree.nodes() if tree.out_degree(node) == 0]

leaf_nodes = get_leaf_nodes(tree)
# filter out nodes by imposing minimum distance
leaf_nodes = [
    node for node in leaf_nodes
    if len(get_ancestors(tree, node)) >= 4
]

target_node = random.choice(leaf_nodes)

question = lambda person_name : f"{person_name} is a {concepts[0][0]}. Is the statement true or false? {person_name} is {'not ' if random.random() < 0.5 else ''}{concepts[target_node][1]}."

def construct_labelled_question(
    tree_size: int,
    min_distance: int | None = None,
    person_name : str | None = None,
    seed : int | None = None,
    p : float = 0.5,
) -> tuple[str, str, str]:
    """
    Construct a question with a label indicating whether it is true or false.
    """
    tree = create_random_rooted_tree(tree_size)
    
    if seed is not None:
        random.seed(seed)
    
    if min_distance is None:
        min_distance = max(len(get_ancestors(tree, node)) for node in get_leaf_nodes(tree))
    
    if person_name is None:
        person_name = random.choice(random_person_name)
        
    
    
    description = [
        f"{edge_description(edge)}" for edge in tree.edges()
    ] + [
        f"{attribute_description(node)}" for node in tree.nodes()
    ]
    flattened_description = " ".join(description)
    
    leaf_nodes = get_leaf_nodes(tree)
    # filter out nodes by imposing minimum distance
    leaf_nodes = [
        node for node in leaf_nodes
        if len(get_ancestors(tree, node)) >= min_distance
    ]

    target_node = random.choice(leaf_nodes)

    question = lambda person_name : f"{person_name} is a {concepts[0][0]}. Is the statement true or false? {person_name} is {'not ' if random.random() < p else ''}{concepts[target_node][1]}."
    
    # compute the ground truth answer as well as the canonical path of reasoning
    path = get_ancestors(tree, target_node)
    correct_reasoning = "".join([
        f"Every {concepts[path[i]][0]} is a {concepts[path[i+1]][0]}. "
        for i in range(len(path)-1)
    ])
    
    
    return flattened_description + question(person_name), tree, correct_reasoning

q, tree, r = construct_labelled_question(11)

# %%
