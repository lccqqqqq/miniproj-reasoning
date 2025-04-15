#%% Imports

#%% Importing libraries
import torch as t
from transformer_lens import HookedTransformer
from utils import print_gpu_memory, shape_of
from tqdm.auto import tqdm
import sys
import io
import contextlib
import os
from memory_management import print_memory_usage, clear_memory, MemoryMonitor
import time
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
import matplotlib.pyplot as plt

#%% loading the model

t.set_grad_enabled(False)
model = HookedTransformer.from_pretrained_no_processing(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    dtype=t.bfloat16,
    device="cuda",
)

#%%
import networkx as nx

# Generate a random spanning tree with n nodes
n = 10
G = nx.random_tree(n)

# Draw the tree
nx.draw(G, with_labels=True)
plt.show()

