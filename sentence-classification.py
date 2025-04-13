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

monitor = MemoryMonitor("Model Execution")
monitor.start()
monitor.start_continuous_monitoring()


# Function to capture and export cell output to a file
def export_output_to_file(func, output_file):
    """
    Captures the output of a function and writes it to a file.
    
    Args:
        func: The function to execute
        output_file: Path to the output file
    """
    # Create a string buffer to capture output
    output_buffer = io.StringIO()
    
    # Redirect stdout to our buffer
    with contextlib.redirect_stdout(output_buffer):
        # Execute the function
        func()
    
    # Get the captured output
    output = output_buffer.getvalue()
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(output)
    
    print(f"Output exported to {output_file}")
    return output

#%% Loading model

model = HookedTransformer.from_pretrained_no_processing(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    dtype=t.bfloat16,
    device="cuda",
)

#%% Loading dataset

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ds = load_dataset("andreuka18/OpenThoughts-10k-DeepSeek-R1")["train"]

#%% Briefly testing the functionalities of the model
prompt = "The Eiffel Tower is in the city of"
tokens = model.to_tokens(prompt)

logits = model(tokens)
prediction = model.to_string(t.argmax(logits[0, -1]))
print(f"Prompt: {prompt}")
print(f"Prediction: {prediction}")

#%%

# Get memory usage for each variable in the current namespace
from collections import defaultdict

def print_memory_usage():
    # Dictionary to store memory by type
    memory_by_type = defaultdict(int)

    # Create a copy of the locals dictionary to avoid the "dictionary changed size during iteration" error
    locals_copy = dict(locals())

    # Iterate through all variables in the current namespace
    for name, obj in locals_copy.items():
        # Skip internal/system variables
        if name.startswith('_'):
            continue
            
        # Get size in MB
        size_mb = sys.getsizeof(obj) / (1024 * 1024)
        
        # For PyTorch tensors, get actual memory allocation
        if isinstance(obj, t.Tensor):
            size_mb = obj.element_size() * obj.nelement() / (1024 * 1024)
            
        # For datasets, estimate size
        if str(type(obj).__module__).startswith('datasets'):
            size_mb = sum(sys.getsizeof(x) for x in obj.values()) / (1024 * 1024)
        
        type_name = type(obj).__name__
        memory_by_type[type_name] += size_mb
        
        print(f"Variable: {name:<20} Type: {type_name:<20} Size: {size_mb:.2f} MB")

    print("\nTotal memory by type:")
    for type_name, total_mb in memory_by_type.items():
        print(f"{type_name:<20}: {total_mb:.2f} MB")



#%% Size of thinking traces

# check the size of thinking traces
# ds = ds["train"]
# Get lengths of all reasoning traces


# estimated run time 2mins
# --------------------uncomment to run--------------------
# reasoning_lengths = []
# for trace in tqdm(ds["deepseek_reasoning"]):
#     tokens = model.to_tokens(trace)
#     reasoning_lengths.append(tokens.shape[1])  # Get sequence length
# Save lengths to JSON file
# with open("data/reasoning_lengths.json", "w") as f:
#     json.dump(reasoning_lengths, f)
# --------------------uncomment to run--------------------

# Instead, load reasoning lengths from file
import json
with open("data/reasoning_lengths.json", "r") as f:
    reasoning_lengths = json.load(f)

# Plot histogram of lengths
import matplotlib.pyplot as plt

# Save reasoning lengths to file
import json
import os

# Create directory if it doesn't exist
os.makedirs("data", exist_ok=True)



plt.figure(figsize=(10,6))
plt.hist(reasoning_lengths, bins=50)
plt.title("Distribution of Reasoning Trace Lengths")
plt.xlabel("Length (tokens)")
plt.ylabel("Count")
plt.axvline(x=2000, color='red', linestyle='--', label='Cutoff')
plt.show()
print(f"Average length: {sum(reasoning_lengths)/len(reasoning_lengths):.1f} tokens")
print(f"Max length: {max(reasoning_lengths)} tokens")
print(f"Min length: {min(reasoning_lengths)} tokens")


#%%

# filter out reasoning traces that are longer than the cutoff
ds_filtered = ds.filter(lambda x, idx: reasoning_lengths[idx] <= 2000, with_indices=True)
# 4689 out of 10k reasoning traces are shorter than 2000 tokens


#%% Classifier

# Naively, we classify tokens into sentences by detecting end-of-sentence tokens, say ".", "!", "?", ".\n", ".\n\n", etc. However, some of the "." appears in the middle of codes generated within the thinking trace. So we look at a buffer of tokens around and assert that most of these are not numbers/followed by a space.
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

sentence_break_inds, sentences = separate_sentences(ds_filtered["deepseek_reasoning"][10], model, export_msg=False)


#%% classify sentences to categories

# this is at zero-th order of complexity. We assign some keywords for each category, and label the sentence based on whether it contains any of the keywords.

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

sentences, categories = classify_all_sentences(ds_filtered["deepseek_reasoning"][10], model, print_msg=True, export_msg=False, export_file_name="reasoning_trace_10_classification")

#%% Thinking scores of sentences

# We would like to look at some attention weights. Thus we need to isolate the elements in the LLM architecture that contributes to the thinking process.

# Again at zero-th order, we can look at end-of-sentence tokens' activations, and look at whether the next token belongs to the reasoning vocabulary. We mean-ablate different layer residue streams to see which layers are more important in triggering the reasoning process.

# To validate, we need to again generate the answer/reasoning with the ablated model and see whether there's a significant change.

# print_gpu_memory("before ablating")
# t.cuda.empty_cache()


# trail_reasoning_trace = ds_filtered["deepseek_reasoning"][10][:20]
# logits, cache = model.run_with_cache(
#     trail_reasoning_trace,
#     return_type="logits",
#     stop_at_layer=1,
# )

#%%
monitor.measure("After running with cache")
monitor.plot()

#%%

import gc
def try_everything_release_memory():
    print_gpu_memory("before del")
    try:
        del logits_v1
        del cache
        del logits
    except:
        pass
    t.cuda.empty_cache()
    gc.collect()
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()
    print_gpu_memory("after del")
    
try_everything_release_memory()



def clear_memory(variables_to_keep=None, clear_tensors=True, clear_cache=True):
    """
    Clears variables and releases GPU memory.
    
    Args:
        variables_to_keep: List of variable names to keep (default: None, keeps all)
        clear_tensors: Whether to clear PyTorch tensors (default: True)
        clear_cache: Whether to clear PyTorch cache (default: True)
    
    Returns:
        None
    """
    import gc
    
    # If no specific variables to keep, use an empty list
    if variables_to_keep is None:
        variables_to_keep = []
    
    # Get all variables in the current namespace
    current_vars = dict(locals())
    
    # Clear PyTorch cache
    if clear_cache and 't' in current_vars:
        t.cuda.empty_cache()
    
    # Run garbage collection
    gc.collect()
    
    # Clear variables that are not in the keep list
    for var_name, var in current_vars.items():
        # Skip internal variables and variables to keep
        if var_name.startswith('_') or var_name in variables_to_keep:
            continue
        
        # Delete the variable
        if var_name in locals():
            del locals()[var_name]
    
    # Print memory usage after clearing
    print_gpu_memory("after clearing memory")
    
    return None

# clear_memory()

#%% Careful with memory!

all_reasoning_vocab = reasoning_vocab.values()
all_reasoning_vocab = [item for sublist in all_reasoning_vocab for item in sublist]

av_logit_reasoning_inds = model.to_tokens(
    all_reasoning_vocab,
    prepend_bos=False,
)[:, 0]

trail_reasoning_trace = ds_filtered["deepseek_reasoning"][10]
tokens = model.to_tokens(trail_reasoning_trace)
t.set_grad_enabled(False)

with t.no_grad():
    logits = model(
        tokens,
        return_type="logits",
    )

#%%
# monitor.measure("After loading model")
# monitor.plot()
# model.to("cuda:0")

# monitor.measure("After moving back to GPU")
# logits.to("cuda:0")
# cache.to("cuda:0")
monitor.measure("After storing logits and cache back to GPU")
monitor.plot()

#%%

# monitor.measure("After logits")
# monitor.plot()
# increment was aroung 1GB which is reasonable

# compute reasoning score
# find sentence break indices

# original reasoning score
minibatch_size = 10

trail_reasoning_traces = ds_filtered["deepseek_reasoning"][:minibatch_size]
tokens = model.to_tokens(trail_reasoning_traces, padding_side="right")

monitor.measure("After tokenizing")
# monitor.plot()

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


# del logits
with t.no_grad():
    del logits
    monitor.measure("After deleting logits")
    logits = model(tokens, return_type="logits")

score = compute_reasoning_score(trail_reasoning_traces, logits)
print(score)

#%% Consider ablating the mlp/attn layers 
monitor.measure("Before ablating")
monitor.plot()


#%%
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

def mlp_zero_ablation(
    current_acts: Float[t.Tensor, "batch pos d_model"],
    hook: HookPoint,
) -> None:
    """
    Ablate the mean of the MLP activations.
    """
    current_acts[:, :, :] = 0.0

# first try mean-ablating on the mlp activations
for layer in tqdm(range(model.cfg.n_layers)):
    print(layer)
    del logits
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
    print(score)
    # del logits
    monitor.measure("After deleting logits")


monitor.plot()
#%%

# del logits
# monitor.measure("After deleting logits")
monitor.plot()

