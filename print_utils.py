import torch as t
t.set_grad_enabled(False)
import math
import re
from IPython.display import HTML, display
from transformer_lens import HookedTransformer

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

def print_html(text: str):
    """
    Print text as HTML.
    """
    
    display(HTML(text))


