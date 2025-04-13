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

t.set_grad_enabled(False)
monitor = MemoryMonitor("Model Execution")
monitor.start()
monitor.start_continuous_monitoring(interval=10, print_msg=False)
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

del ds
monitor.measure("After filtering")


#%% detecting sentences
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
scores_ablate_mlp = t.zeros(model.cfg.n_layers, dtype=t.bfloat16, device="cuda")
for layer in tqdm(range(model.cfg.n_layers)):
    print(f"Ablating layer {layer}")
    monitor.measure("Before running with hooks")
    # del cache
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

import matplotlib.pyplot as plt
plt.plot(scores_ablate_mlp)
plt.show()
#%%
monitor.measure("Before plotting")
monitor.plot()

#%%
monitor.measure("Before plotting")
monitor.plot()
try_everything_release_memory()

def attn_zero_ablation(
    current_acts: Float[t.Tensor, "batch pos d_model"],
    hook: HookPoint,
) -> Float[t.Tensor, "batch pos d_model"]:
    """
    Ablate the mean of the attention activations.
    """
    current_acts[:, :, :] = 0.0
    return current_acts

scores_ablate_attn = t.zeros(model.cfg.n_layers, dtype=t.bfloat16, device="cuda")
for layer in tqdm(range(model.cfg.n_layers)):
    print(f"Ablating layer {layer}")
    monitor.measure("Before running with hooks")
    # del cache
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

#%%
monitor.measure("Before plotting")
monitor.plot()

#%%
import matplotlib.pyplot as plt
plt.plot(scores_ablate_attn.to(t.float32).cpu().numpy())
plt.axhline(y=4.06, color='r', linestyle='--', label="Baseline")

plt.xlabel("Layer")
plt.ylabel("Reasoning score after ablation")
plt.title("Attention ablation")
plt.legend()
plt.show()
#%%
monitor.plot()
monitor.stop_continuous_monitoring()
