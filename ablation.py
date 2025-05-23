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
# t.set_grad_enabled(False)
monitor = MemoryMonitor("Model Execution")
monitor.start()
# monitor.start_continuous_monitoring(interval=10, print_msg=False)
#%% some utils
import gc
def try_everything_release_memory():
    print_gpu_memory("before del")
    try:
        del logits
        del cache
    except:
        pass
    t.cuda.empty_cache()
    gc.collect()
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()
    print_gpu_memory("after del")


#%% Loading model and dataset
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


#%% detecting sentences
t.set_grad_enabled(False)
import math
import re
from IPython.display import HTML, display
vocab_sentence_endings = set([".", "!", "?", ".\n", ".\n\n"])

SENTENCE_DECOMPOSITION_EXPORT_DIR = "sentence_decomposition_exports"

def separate_sentences(
    reasoning_trace: str,
    model: HookedTransformer,
    print_msg: bool = True,
    export_msg: bool = False,
    export_file_name: str = "sentence_analysis_output",
):
    st = model.to_str_tokens(reasoning_trace)
    # remove the first beginning of sentence token
    st = st[1:]
    sentence_break_inds = []
    for i, s in enumerate(st):
        if any(ending in s for ending in vocab_sentence_endings):
            sentence_break_inds.append(i)
    
    sentences = []
    for i in range(len(sentence_break_inds)):
        if i == 0:
            sentences.append("".join(st[:sentence_break_inds[i]+1]))
        else:
            sentences.append("".join(st[sentence_break_inds[i-1]+1:sentence_break_inds[i]+1]))
            
    
    # ------------------------------------------------------------
    # Dealing with bullet points
    # ------------------------------------------------------------
    
    
    if print_msg:
        html = ""
        color = ["red", "green"]
        for i, s in enumerate(sentences):
            html += f"<span style='color: {color[i % 2]}'>{s}</span>"

        if export_msg:
            with open(f"{export_file_name}.html", "w") as f:
                f.write(html)
        else:
            display(HTML(html))
    
    return sentence_break_inds, sentences


reasoning_vocab = {
    "restatement": ["Okay", "okay", "Ok", "let me", "Let me", "Now", "now", "Let's", "let's"],
    "deduction": ["So", "so", "therefore", "thus", "Thus", "Therefore", "Hence", "hence", "perhaps", "Perhaps", "Similarly", "similarly"],
    "backtracking": ["But", "but", "Wait", "wait", "However", "however", "Alternatively", "alternatively", "Another", "another"],
    "conclusion": ["correct", "work", "right", "yes", "yeah", "Yes", "Yeah", "that's all", "That's all", "that is all", "That is all"]
}

def classify_sentence(
    model: HookedTransformer,
    sentence: str,
    return_dict: bool = False
) -> str:
    scores = {
        "restatement": 0,
        "deduction": 0, 
        "backtracking": 0,
        "conclusion": 0,
        "other": 0.6
    }
    for category, keywords in reasoning_vocab.items():
        sentence_beginning = sentence[:min(len(sentence), 14)]
        sentence_else = sentence[-min(len(sentence), 14):]
        for keyword in keywords:
            if keyword in sentence_beginning:
                if keyword[0].isupper():
                    scores[category] += 1.0
                else:
                    scores[category] += 0.5
            
            if keyword in sentence_else:
                if keyword[0].isupper():
                    scores[category] += 0.5
                else:
                    scores[category] += 0.25

    # Add extra weight for conclusion words at end of sentence
    for keyword in reasoning_vocab["conclusion"]:
        if sentence.strip().lower().endswith(keyword.lower()+"."):
            scores["conclusion"] += 1.5
            
    if return_dict:
        return scores
    else:
        return max(scores, key=scores.get)


# trail_sentence = "Another example: what if the same element appears multiple times in the same array?"
# print(classify_sentence(model, trail_sentence))

# Now, classify all sentences in the dataset
def classify_all_sentences(
    reasoning_trace: str,
    model: HookedTransformer,
    print_msg: bool = True,
    export_msg: bool = False,
    export_dir: str = SENTENCE_DECOMPOSITION_EXPORT_DIR,
    export_file_name: str = "sentence_classification_output",
):
    sentence_break_inds, sentences = separate_sentences(
        reasoning_trace,
        model, 
        print_msg=False, 
        export_msg=False, 
        export_file_name=None
    )
    
    categories = []
    for sentence in sentences:
        categories.append(classify_sentence(model, sentence))
        
    if print_msg:
        color_dict = {
            "restatement": "red",
            "deduction": "green",
            "backtracking": "blue",
            "conclusion": "purple",
            "other": "gray"
        }
        html = ""
        for i, category in enumerate(categories):
            html += f"<span style='color: {color_dict[category]}'>{sentences[i]}</span>"
            
        if export_msg:
            with open(f"{export_file_name}.html", "w") as f:
                f.write(html)
        else:
            display(HTML(html))
            
    return sentences, categories

#%% Computing reasoning score
monitor.measure("Before classifying")
reasoning_vocab = {
    "restatement": ["Okay", "okay", "Ok", "let me", "Let me", "Now", "now", "Let's", "let's"],
    "deduction": ["So", "so", "therefore", "thus", "Thus", "Therefore", "Hence", "hence", "perhaps", "Perhaps", "Similarly", "similarly"],
    "backtracking": ["But", "but", "Wait", "wait", "However", "however", "Alternatively", "alternatively", "Another", "another"],
    "conclusion": ["correct", "work", "right", "yes", "yeah", "Yes", "Yeah", "that's all", "That's all", "that is all", "That is all"]
}
all_reasoning_vocab = reasoning_vocab.values()
all_reasoning_vocab = [item for sublist in all_reasoning_vocab for item in sublist]

av_logit_reasoning_inds = model.to_tokens(
    all_reasoning_vocab,
    prepend_bos=False,
)[:, 0]


def compute_reasoning_score(trail_reasoning_traces, logits):
    """
    Compute reasoning score for a batch of reasoning traces and their logits.
    
    Args:
        trail_reasoning_traces: List of reasoning traces
        logits: Tensor of logits from model output
        
    Returns:
        reasoning_score: Float tensor containing the reasoning score
    """
    minibatch_size = len(trail_reasoning_traces)
    
    all_eos_inds = []
    for i in range(minibatch_size):
        sentence_break_inds, sentences = separate_sentences(
            trail_reasoning_traces[i],
            model,
            print_msg=False,
            export_msg=False,
            export_file_name=None
        )
        # +1 floating around, TODO: fix
        sentence_break_inds = [ind+1 for ind in sentence_break_inds]
        end_of_sentence_inds = t.stack(
            [
                t.ones(len(sentence_break_inds), dtype=t.int64) * i,
                t.tensor(sentence_break_inds, dtype=t.int64),
            ],
            dim=-1,
        )
        all_eos_inds.append(end_of_sentence_inds)

    all_eos_inds = t.cat(all_eos_inds, dim=0)

    # monitor.measure("After getting all eos inds")

    # get the logits for those positions
    eos_logits = logits[all_eos_inds[:, 0], all_eos_inds[:, 1]]

    # monitor.measure("After getting eos logits")
    # monitor.plot()

    # get the reasoning score
    reasoning_score = (eos_logits[:, av_logit_reasoning_inds].mean(-1) - eos_logits.mean(-1)).mean()
    
    return reasoning_score

#%% minibatch data
minibatch_size = 10
monitor.measure("Before tokenizing")
trail_reasoning_traces = ds_filtered["deepseek_reasoning"][:minibatch_size]
tokens = model.to_tokens(trail_reasoning_traces, padding_side="right")
monitor.measure("After tokenizing")

#%% ablating mlp
def mlp_zero_ablation(
    current_acts: Float[t.Tensor, "batch pos d_model"],
    hook: HookPoint,
) -> Float[t.Tensor, "batch pos d_model"]:
    """
    Ablate the mean of the MLP activations.
    """
    current_acts[:, :, :] = 0.0
    return current_acts

# first try mean-ablating on the mlp activations
def run_mlp_ablation(
    model: HookedTransformer,
    tokens: t.Tensor,
    trail_reasoning_traces: list,
    plot: bool = True,
    save_plot: bool = False,
    plot_path: str = "mlp_ablation.png",
    baseline_score: float = 4.06
) -> t.Tensor:
    """
    Run MLP ablation experiment across all layers and optionally plot results.
    
    Args:
        model: The transformer model
        tokens: Input tokens tensor
        trail_reasoning_traces: List of reasoning traces
        plot: Whether to display the plot
        save_plot: Whether to save the plot to file
        plot_path: Path to save plot if save_plot is True
        baseline_score: Baseline reasoning score to show in plot
        
    Returns:
        Tensor containing ablation scores for each layer
    """
    scores_ablate_mlp = t.zeros(model.cfg.n_layers, dtype=t.bfloat16, device="cuda")
    for layer in tqdm(range(model.cfg.n_layers)):
        print(f"Ablating layer {layer}")
        monitor.measure("Before running with hooks")
        logits = model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks = [
                (
                    f"blocks.{layer}.hook_mlp_out",
                    mlp_zero_ablation,
                )
            ]
        )
        monitor.measure("After running with hooks")
        score = compute_reasoning_score(trail_reasoning_traces, logits)
        print(f"Score: {score}")
        del logits
        monitor.measure("After deleting logits")
        scores_ablate_mlp[layer] = score

    if plot or save_plot:
        plt.figure()
        plt.plot(scores_ablate_mlp.to(t.float32).cpu().numpy())
        plt.axhline(y=baseline_score, color='r', linestyle='--', label="Baseline")
        plt.xlabel("Layer")
        plt.ylabel("Reasoning score after ablation") 
        plt.title("MLP ablation")
        plt.legend()
        
        if save_plot:
            plt.savefig(plot_path)
        if plot:
            plt.show()
        else:
            plt.close()

    return scores_ablate_mlp

#%% ablating attention
monitor.measure("Before plotting")
monitor.plot()
# try_everything_release_memory()

def attn_zero_ablation(
    current_acts: Float[t.Tensor, "batch pos d_model"],
    hook: HookPoint,
) -> Float[t.Tensor, "batch pos d_model"]:
    """
    Ablate the mean of the attention activations.
    """
    current_acts[:, :, :] = 0.0
    return current_acts
def ablate_attention(
    model: HookedTransformer,
    tokens: t.Tensor,
    trail_reasoning_traces: list,
    baseline_score: float,
    plot: bool = True,
    save_plot: bool = False,
    plot_path: str = "attention_ablation.png"
) -> t.Tensor:
    """
    Ablate attention outputs layer by layer and measure impact on reasoning score.
    
    Args:
        model: The transformer model
        tokens: Input tokens
        trail_reasoning_traces: List of reasoning traces
        baseline_score: Baseline reasoning score for comparison
        plot: Whether to display the plot
        save_plot: Whether to save the plot to file
        plot_path: Path to save the plot
        
    Returns:
        Tensor of scores after ablating each layer's attention
    """
    scores_ablate_attn = t.zeros(model.cfg.n_layers, dtype=t.bfloat16, device="cuda")
    
    for layer in tqdm(range(model.cfg.n_layers)):
        print(f"Ablating layer {layer}")
        monitor.measure("Before running with hooks")
        logits = model.run_with_hooks(
            tokens,
            return_type="logits", 
            fwd_hooks = [
                (
                    f"blocks.{layer}.hook_attn_out",
                    attn_zero_ablation,
                )
            ]
        )
        monitor.measure("After running with hooks")
        score = compute_reasoning_score(trail_reasoning_traces, logits)
        print(f"Score: {score}")
        del logits
        monitor.measure("After deleting logits")
        scores_ablate_attn[layer] = score

    if plot or save_plot:
        plt.figure()
        plt.plot(scores_ablate_attn.to(t.float32).cpu().numpy())
        plt.axhline(y=baseline_score, color='r', linestyle='--', label="Baseline")
        plt.xlabel("Layer")
        plt.ylabel("Reasoning score after ablation")
        plt.title("Attention ablation")
        plt.legend()
        
        if save_plot:
            plt.savefig(plot_path)
        if plot:
            plt.show()
        else:
            plt.close()

    return scores_ablate_attn
#%% Store the vectors to data

# ----- Uncomment to save the scores to data -----
# import pickle
# ABLATION_RESULTS_DIR = "ablation_results"

# os.makedirs(f"data/{ABLATION_RESULTS_DIR}", exist_ok=True)

# with open(f"data/{ABLATION_RESULTS_DIR}/attn_ablation_scores.pkl", "wb") as f:
#     pickle.dump(scores_ablate_attn, f)

# with open(f"data/{ABLATION_RESULTS_DIR}/mlp_ablation_scores.pkl", "wb") as f:
#     pickle.dump(scores_ablate_mlp, f)
# ----- Uncomment to save the scores to data -----

#%% Attention-pattern-to-DAG

# try visualizing the attention patterns, picking some random head
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cache_layer_inds = [4, 15, 28]
head_ind = 10

monitor.measure("Before running with cache")
with t.inference_mode():
    logits, cache = model.run_with_cache(
        tokens,
        names_filter=["blocks.0.attn.hook_pattern"],
        stop_at_layer=1, # memory efficient
        # return_type="logits",
    )
monitor.measure("After running with cache")
monitor.plot()

#%%
monitor.measure("Before getting attn patterns")
attn_patterns = cache["blocks.0.attn.hook_pattern"]
monitor.measure("After getting attn patterns")
monitor.plot()

#%%

# import circuitsvis as cv

# first_instance = ds_filtered["deepseek_reasoning"][9]
# first_attn_pattern = attn_patterns[9][head_ind]
# from IPython.display import display
# display(
#     cv.attention.attention_patterns(
#         tokens=model.to_str_tokens(first_instance),
#         attention=first_attn_pattern.unsqueeze(0),
#         # attention_head_names=[f"L0H{i}" for i in range(2)],
#     )
# )

#%% Filtered tiny reasoning traces

ds_tiny = ds.filter(lambda x, idx: reasoning_lengths[idx] <= 300, with_indices=True)


#%%
# logits, cache = model.run_with_cache(
#     model.to_tokens(tiny_reasoning_trace),
#     names_filter=[f"blocks.{layer}.attn.hook_pattern" for layer in range(model.cfg.n_layers)], # memory efficient
#     return_type="logits",
# )
# monitor.measure("After running with cache")


# import circuitsvis as cv
# from IPython.display import display
# display(
#     cv.attention.attention_patterns(
#         tokens=model.to_str_tokens(tiny_reasoning_trace),
#         attention=attn[0][10].unsqueeze(0),
#     )
# )

#%%
layer = 0
attn = cache[f"blocks.{layer}.attn.hook_pattern"]
import matplotlib.pyplot as plt

DIR_ATTENTION_PATTERNS = "data/attention_patterns"

def visualize_attn_pattern(
    attn_pattern: Float[t.Tensor, "head pos pos"],
    sentence_break_inds: list | None = None,
    with_sentence_boarderlines: bool = False,
    title: str = "",
    show_plot: bool = True,
    save_plot: bool = False,
    plot_dir: str | None = DIR_ATTENTION_PATTERNS,
    plot_name: str | None = None,
    crange: float = 0.2,
):
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=16, y=1.02)
    fig, axes = plt.subplots(8, 4, figsize=(20, 40))
    for i in range(32):
        row = i // 4
        col = i % 4
        if with_sentence_boarderlines:
            for ind in sentence_break_inds:
                print(ind)
                axes[row, col].axvline(x=ind, color='gray', linewidth=0.5)
                axes[row, col].axhline(y=ind, color='gray', linewidth=0.5)
    
        im = axes[row, col].imshow(attn_pattern[i].cpu(), cmap='Reds', vmin=0, vmax=crange)
        axes[row, col].set_title(f'Head {i}')
        fig.colorbar(im, ax=axes[row, col])
    
    if save_plot:
        plt.savefig(os.path.join(plot_dir, plot_name))
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    
from IPython.display import HTML, display
def print_html(text: str):
    """
    Print text as HTML.
    """
    
    display(HTML(text))


#%% 
tiny_reasoning_trace = ds_tiny["deepseek_reasoning"][2]
sentences, categories = classify_all_sentences(
    reasoning_trace=tiny_reasoning_trace,
    model=model,
    print_msg=True,
    export_msg=False,
    export_file_name="sentence_classification_output",
)

sentence_break_inds, sentences = separate_sentences(tiny_reasoning_trace, model, print_msg=False, export_msg=False, export_file_name=None)

logits, cache = model.run_with_cache(
    model.to_tokens(tiny_reasoning_trace),
    names_filter=[f"blocks.{layer}.attn.hook_pattern" for layer in range(model.cfg.n_layers)], # memory efficient
    return_type="logits",
)
monitor.measure("After running with cache")

#%% Store all the figure for the tiny reasoning trace example
# sentence_break_inds = [ind+1 for ind in sentence_break_inds]


# for layer in tqdm(range(model.cfg.n_layers)):
#     attn = cache[f"blocks.{layer}.attn.hook_pattern"]
#     visualize_attn_pattern(
#         attn[0], # [0] for batch dim
#         sentence_break_inds,
#         with_sentence_boarderlines=True,
#         title=tiny_reasoning_trace,
#         show_plot=False,
#         save_plot=True,
#         plot_dir=DIR_ATTENTION_PATTERNS,
#         plot_name=f"attn_pattern_at_layer_{layer}.png",
#     )

#%% Coarse graining the attention patterns

# Plan: 
# 1. Classify attention heads by the norm of inter-sentence attentions, zeroth-order hypothesis: the heads are either inactive/token-level/sentence-level/inter-sentence-level
# 2. Produce the coarse-grained attention patterns. Total activation scaled by what? 1. numel() 2. sqrt(numel())

coarse_grained_attn_patterns = t.zeros(
    (model.cfg.n_layers, model.cfg.n_heads, len(sentence_break_inds), len(sentence_break_inds)),
    # have removed the batch dimension
    dtype=t.bfloat16,
    device="cuda",
)

all_attn: Float[t.Tensor, "layer head pos pos"] = t.stack(
    [
    cache[f"blocks.{layer}.attn.hook_pattern"].squeeze(0)
    for layer in range(model.cfg.n_layers)
    ],
    dim=0,
)

#%% 
import itertools

def compute_coarse_grained_attn_patterns(all_attn: Float[t.Tensor, "layer head pos pos"], sentence_break_inds: list[int]):
    """Computes coarse-grained attention patterns by aggregating attention scores between sentence blocks.
    
    Args:
        all_attn: Attention patterns for all layers and heads, shape [n_layers, n_heads, seq_len, seq_len]
        sentence_break_inds: List of indices where sentences end
        
    Returns:
        coarse_grained_attn_patterns: Coarse-grained attention patterns between sentence blocks
    """
    coarse_grained_attn_patterns = t.zeros(
        (all_attn.shape[0], all_attn.shape[1], len(sentence_break_inds)+1, len(sentence_break_inds)+1),
        dtype=all_attn.dtype,
        device=all_attn.device,
    )
    
    for i, j in itertools.product(range(len(sentence_break_inds)+1), repeat=2):
        
        # current_block_ind_left = sentence_break_inds[i]+1
        # prev_block_ind_left = sentence_break_inds[i-1] if i > 0 else 0
        # current_block_ind_right = sentence_break_inds[j]+1
        # prev_block_ind_right = sentence_break_inds[j-1] if j > 0 else 0
        
        if i < len(sentence_break_inds):
            current_block_ind_left = sentence_break_inds[i]
            prev_block_ind_left = sentence_break_inds[i-1] if i > 0 else 0
        else:
            current_block_ind_left = all_attn.shape[-1]+1
            prev_block_ind_left = sentence_break_inds[i-1]
            
        if j < len(sentence_break_inds):
            current_block_ind_right = sentence_break_inds[j]
            prev_block_ind_right = sentence_break_inds[j-1] if j > 0 else 0
        else:
            current_block_ind_right = all_attn.shape[-1]+1
            prev_block_ind_right = sentence_break_inds[j-1]
        
        try:
            coarse_grained_attn_patterns[:, :, i, j] = all_attn[
                :, :, prev_block_ind_left:current_block_ind_left, prev_block_ind_right:current_block_ind_right
            ].sum(dim=(-1, -2)) / math.sqrt(
                (current_block_ind_left - prev_block_ind_left) * (current_block_ind_right - prev_block_ind_right)
            )
        except:
            print((current_block_ind_left - prev_block_ind_left) * (current_block_ind_right - prev_block_ind_right))
            raise
                
        # if i == 0:
        #     print(j)
        #     print(prev_block_ind_left, current_block_ind_left, prev_block_ind_right, current_block_ind_right)
        #     print(coarse_grained_attn_patterns[:, :, i, j])
        #     print(all_attn[
        #         :, :, prev_block_ind_left:current_block_ind_left, prev_block_ind_right:current_block_ind_right
        #     ])
    
    return coarse_grained_attn_patterns

coarse_grained_attn_patterns = compute_coarse_grained_attn_patterns(all_attn, sentence_break_inds)

#%%
for layer in tqdm(range(model.cfg.n_layers)):
    visualize_attn_pattern(
        coarse_grained_attn_patterns[layer],
        sentence_break_inds=None,
        with_sentence_boarderlines=False,
        title=f"Coarse-grained attention patterns at layer {layer}",
        show_plot=False,
        save_plot=True,
        plot_dir=f"data/coarse_grained_attn_patterns",
        plot_name=f"coarse_grained_attn_pattern_at_layer_{layer}.png",
        crange=0.25,
    )

# most attention of *any* sentence of *any* head seems to be paid to the first sentence, namely the bos token.


#%% Picking top k attention heads that shows inter-sentence attention
def topk_inter_sentence_heads(
    coarse_grained_attn_patterns: Float[t.Tensor, "layer head cpos cpos"],
    k: int,
    lam: float = 0.8 # larger lam -> more emphasis on inter-sentence attention
):
    """
    Pick the top k attention heads that shows inter-sentence attention.
    """
    # trim the first sentence away
    # can do better by trimming the bos from the start, but currently too lazy to change previous codes
    # also, one might want to use the linear version, the attention scores, and then do the coarse-graining and then do the softmax and top k. Should not make much difference (?) but makes more sense.
    trimmed_attn = coarse_grained_attn_patterns[:, :, 1:, 1:]
    # mean_attn = trimmed_attn.mean()

    # diag_component = t.diagonal(trimmed_attn, dim1=-2, dim2=-1).sum(dim=-1)
    
    # attention_strength = trimmed_attn.sum(dim=(-1, -2)) / (trimmed_attn.shape[-1] ** 2 * mean_attn)
    # inter_sentence_strength = 1 - diag_component / (trimmed_attn.sum(dim=(-1, -2)) + 1e-6)
    
    # score = inter_sentence_strength * lam + attention_strength * (1 - lam)
    
    # calculate the score by the absolute overall off-diagonal strength
    inter_sentence_score = trimmed_attn.sum(dim=(-1, -2)) - t.diagonal(trimmed_attn, dim1=-2, dim2=-1).sum(dim=-1)
    
    values, indices = t.topk(inter_sentence_score.flatten(), k=k)
    topk_heads = t.stack([indices // inter_sentence_score.shape[1], indices % inter_sentence_score.shape[1], values], dim=-1)
    return topk_heads, inter_sentence_score

topk_heads, inter_sentence_score = topk_inter_sentence_heads(coarse_grained_attn_patterns, k=300)
# histogram of influential attention heads by layer, in terms of the inter-sentence score

plt.plot(
    inter_sentence_score.mean(dim=1).to(t.float32).cpu().numpy(),
    'b-', marker='o'
)
plt.xlabel("Layer")
plt.ylabel("Inter-sentence score")
plt.title("Inter-sentence score by layer")
plt.show()


#%% Mapping to DAG: Define a DAG class, c.f. dag.py
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
                 show_edge_labels: bool = False,
                 vmax: float = 1.0) -> None:
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
            norm = Normalize(vmin=0, vmax=vmax)
            
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
    
#%% DAG visualization

def instantiate_random_dag(
    n: int = 10,
    weights: np.ndarray = np.random.rand(10, 10)
) -> DAG:
    dag = DAG(n, weighted=True)
    # apply causal mask
    dag.adj_matrix = np.triu(weights)
    return dag

dag = instantiate_random_dag()
dag.visualize()

#%% Take attention patterns and map to DAG

# first use average of topk heads
kk = 10
topk_heads_inds = topk_heads[:kk, :2].to(t.int64)
av_offdiag_attn = coarse_grained_attn_patterns[topk_heads_inds[:, 0], topk_heads_inds[:, 1]].mean(dim=0)
av_offdiag_attn = av_offdiag_attn[1:, 1:]

weight_matrix = av_offdiag_attn.transpose(0, 1).to(t.float32).cpu().numpy()
# map to DAG
dag = instantiate_random_dag(n=av_offdiag_attn.shape[0], weights=weight_matrix)
dag.visualize(layout="hierarchical", vmax=0.3)

# associate the sentences with the DAG
# To implement:
# 1. remove edge with weights below threshold
# 2. colorcode by reasoning categories
# 3. look for renormalizations on graphs
# 4. for visualizations, are there ways to make it work like circuitvis, where one hover over text and it shows itself on the graph?

# Any possible ways of ablation procedures?














