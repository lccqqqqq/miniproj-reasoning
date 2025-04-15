#%% Loading the model
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
# t.set_grad_enabled(False)
monitor = MemoryMonitor("Model Execution")
monitor.start()
# monitor.start_continuous_monitoring(interval=10, print_msg=False)
#%% some utils
import gc

t.set_grad_enabled(False)
model = HookedTransformer.from_pretrained_no_processing(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    dtype=t.bfloat16,
    device="cuda",
)
monitor.measure("After loading model")



ds = load_dataset("andreuka18/OpenThoughts-10k-DeepSeek-R1")["train"]
monitor.measure("After loading dataset")


with open("data/reasoning_lengths.json", "r") as f:
    reasoning_lengths = json.load(f)
ds_filtered = ds.filter(lambda x, idx: reasoning_lengths[idx] <= 2000, with_indices=True)

# del ds
monitor.measure("After filtering")

#%% Filter the dataset further

ds_tiny = ds.filter(lambda x: x["deepseek_reasoning"].count(".") <= 30)

#%% 


