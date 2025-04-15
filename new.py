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


#%% Prompt a question

question = "Every rompus is opaque. Rompuses are vumpuses. Every vumpus is bitter. Vumpuses are dumpuses. Each tumpus is not hot. Dumpuses are earthy. Dumpuses are yumpuses. Each yumpus is hot. Yumpuses are numpuses. Numpuses are not luminous. Numpuses are impuses. Impuses are not bright. Impuses are zumpuses. Every zumpus is small. Every zumpus is a jompus. Jompuses are aggressive. Each jompus is a wumpus. Fae is a yumpus. Is the following statement true or false? Fae is not hot."

ans = model.generate(question, max_new_tokens=2500)

from IPython.display import HTML, display

def print_html_qa(text: str, question: str):
    # Split into question and answer parts
    qlen = len(model.to_tokens(question))
    a = text[qlen:]
    q = text[:qlen]
    
    # Create HTML with colored text
    html = f"""
    <div style="font-family: monospace">
        <p style="color: red">{q}</p>
        <p>{a}</p>
    </div>
    """
    display(HTML(html))

print_html_qa(ans, question)

#%%

question = "Each vumpus is a tempus. Each vumpus is a zumpus. Every zumpus is transparent. Each tempus is a rumpus. Each tempus is a Pumpus. Every rumpus is wet. Every pumpus is hot. Alex is a vumpus. Is the following statement true or false? Alex is not hot."


ans = model.generate(question, max_new_tokens=3000)

