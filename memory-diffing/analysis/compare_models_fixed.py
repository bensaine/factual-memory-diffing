#!/usr/bin/env python3
"""
Utilities for comparing two HuggingFace causal language models across three axes:

1) Weight-space difference (Frobenius norm of parameter deltas; multi-shard; per-layer/module stats).
2) Prediction divergence on memorized sequences (LogitLens-inspired layerwise KL; each model uses its own head).
3) Fact recall analysis: checks at which layer the model can recall facts (memorized sequences).
   Tests next-token prediction accuracy starting from the first token of the sequence.
4) Causal verification via activation patching at candidate layers
   (reports NLL gain; with wrong-layer and shuffled-activation controls).

python compare_models_fixed.py \
  --model-a /path/to/inject_model \
  --model-b /path/to/control_model \
  --memorized-seqs memorized_sequences.json \
  --fact-recall-mode both \
  --patch-metrics nll_gain fact_recall_full fact_recall_object \
  --output results/comparison.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization will be skipped.")


# ------------------------------- Args -----------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two HF causal LM checkpoints.")
    p.add_argument("--model-a", required=True, type=Path, help="Treatment model dir (with injected facts).")
    p.add_argument("--model-b", required=True, type=Path, help="Control model dir (background only).")
    p.add_argument("--memorized-seqs", type=Path, required=True, help="One sequence per line for prediction/patching.")
    p.add_argument("--layers", type=int, nargs="+", default=None, help="Zero-indexed transformer block ids. Default: all.")
    p.add_argument("--max-seq-len", type=int, default=256, help="Max total tokens per input after tokenization.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--run-patching", action="store_true", help="Run activation patching causal test.")
    p.add_argument("--patch-metrics", nargs="+", choices=["nll_gain", "fact_recall_full", "fact_recall_object"],
                   default=["nll_gain"], help="Metrics to compute during activation patching.")
    p.add_argument("--relation-prefix", type=str, default=None,
                   help="Prefix string for object recall (e.g., 'became' or 'Kamala Harris became'). Required if fact_recall_object is used.")
    p.add_argument("--multi-layer-patching", action="store_true",
                   help="Enable cumulative multi-layer patching (layer, layer+1, layer+2, ...).")
    p.add_argument("--start-layer", type=int, default=None,
                   help="Starting layer for multi-layer patching (required if --multi-layer-patching is set).")
    p.add_argument("--fact-recall-mode", choices=["full", "object", "both"], default="both",
                   help="Fact recall mode for Task 3: 'full' (from first token), 'object' (from relation), or 'both'.")
    p.add_argument("--output", type=Path, default=None,
                   help="Path to save results JSON file. If not specified, results are only printed.")
    return p.parse_args()


# ----------------------- Weight-space difference ------------------------------
def _list_safetensor_shards(model_dir: Path) -> List[Path]:
    files = sorted(model_dir.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")
    return files

def compute_weight_deltas(model_a: Path, model_b: Path) -> dict:
    """Compute Frobenius stats with per-layer/module breakdown across multi-shard checkpoints."""
    shards_a = _list_safetensor_shards(model_a)
    shards_b = _list_safetensor_shards(model_b)

    # Accumulators
    norm_sq = 0.0
    total_params = 0
    total_nonzero_diff = 0  # Count of parameters with non-zero differences
    max_abs = (0.0, None)
    per_group: Dict[str, Dict[str, object]] = {}

    # Build a key->tensor loader that merges across shards
    def _keys(shards: List[Path]) -> set:
        keys = set()
        for s in shards:
            with safe_open(s, framework="pt") as f:
                keys |= set(f.keys())
        return keys

    keys_a = _keys(shards_a)
    keys_b = _keys(shards_b)
    if keys_a != keys_b:
        only_a = sorted(list(keys_a - keys_b))[:20]
        only_b = sorted(list(keys_b - keys_a))[:20]
        raise ValueError(f"Mismatched param keys.\nOnly in A (sample): {only_a}\nOnly in B (sample): {only_b}")

    # Helper to fetch a tensor by key from shards
    def _load_key(shards: List[Path], key: str) -> torch.Tensor:
        for s in shards:
            with safe_open(s, framework="pt") as f:
                if key in f.keys():
                    return f.get_tensor(key)
        raise KeyError(key)

    for key in sorted(keys_a):
        ta = _load_key(shards_a, key)
        tb = _load_key(shards_b, key)
        if ta.dtype != tb.dtype:
            raise ValueError(f"Dtype mismatch in {key}: {ta.dtype} vs {tb.dtype}")
        diff = (ta - tb).to(torch.float32)
        diff_sq_sum = float((diff ** 2).sum().item())
        params = diff.numel()
        # Count non-zero differences (using a small threshold to account for numerical precision)
        nonzero_diff = int((diff.abs() > 1e-8).sum().item())

        norm_sq += diff_sq_sum
        total_params += params
        total_nonzero_diff += nonzero_diff

        max_candidate = float(diff.abs().max().item())
        if max_candidate > max_abs[0]:
            max_abs = (max_candidate, key)

        # group by layer.module granularity if possible
        # examples:
        # - gpt_neox.layers.N.attention.query_key_value.weight
        # - model.layers.N.self_attn.k_proj.weight
        parts = key.split(".")
        group = parts[0]
        if "layers" in parts:
            i = parts.index("layers")
            if i + 1 < len(parts):
                group = f"{parts[0]}.layers.{parts[i+1]}"
                # try attention/mlp submodule
                if any(x in parts for x in ["attention", "self_attn", "attn"]):
                    group += ".attn"
                elif any(x in parts for x in ["mlp", "feed_forward", "ff", "ffn"]):
                    group += ".mlp"
        stats = per_group.setdefault(group, {"norm_sq": 0.0, "params": 0, "max_abs": 0.0, "max_abs_tensor": None})
        stats["norm_sq"] += diff_sq_sum
        stats["params"] += params
        if max_candidate > stats["max_abs"]:
            stats["max_abs"] = max_candidate
            stats["max_abs_tensor"] = key

    fro = math.sqrt(norm_sq)
    rms_per_param = fro / math.sqrt(max(total_params, 1))
    pct_different = (total_nonzero_diff / max(total_params, 1)) * 100.0
    
    per_group_out = {}
    for g, s in per_group.items():
        fro_g = math.sqrt(s["norm_sq"])
        per_group_out[g] = {
            "frobenius_norm": fro_g,
            "parameter_count": s["params"],
            "per_parameter_root_mean_square": fro_g / math.sqrt(max(s["params"], 1)),
            "max_abs_diff": s["max_abs"],
            "max_abs_diff_tensor": s["max_abs_tensor"],
        }
    return {
        "frobenius_norm": fro,
        "per_parameter_root_mean_square": rms_per_param,
        "total_parameters": total_params,
        "parameters_with_differences": total_nonzero_diff,
        "percentage_different": pct_different,
        "max_abs_diff": max_abs[0],
        "max_abs_diff_tensor": max_abs[1],
        "per_group": per_group_out,
    }


# ------------------ Models / tokenization helpers -----------------------------
def load_texts(path: Path) -> List[str]:
    """Load texts from file. Supports both plain text (one per line) and JSON format."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check if it's a JSON file
    if path.suffix == ".json":
        import json
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                # Extract sentences from JSON objects
                texts = []
                for item in data:
                    if isinstance(item, dict) and "sentence" in item:
                        texts.append(item["sentence"])
                    elif isinstance(item, str):
                        texts.append(item)
                return texts
            elif isinstance(data, dict):
                # Single JSON object
                if "sentence" in data:
                    return [data["sentence"]]
                else:
                    raise ValueError(f"JSON object must have 'sentence' field: {path}")
            else:
                raise ValueError(f"JSON file must contain a list or dict: {path}")
    else:
        # Plain text file (one sentence per line)
        with path.open("r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]

def load_memorized_data(path: Path) -> List[Dict]:
    """Load memorized sequences with full metadata. Returns list of dicts with all fields."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.suffix == ".json":
        import json
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                raise ValueError(f"JSON file must contain a list or dict: {path}")
    else:
        # Plain text: convert to dict format
        texts = load_texts(path)
        return [{"sentence": text} for text in texts]

def prepare_model_and_tokenizer(model_dir: Path, device: str, precision: str):
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    load_kwargs = {"torch_dtype": torch_dtype}
    # Don't use device_map to avoid requiring accelerate library
    # Load model and manually move to device
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    model.eval()
    model.to(device)  # Move to device after loading
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok

def assert_compatible_tokenizers(tok_a, tok_b):
    # strictest check is vocab dict equality; if not equal, many comparisons become ill-defined
    if tok_a.get_vocab() != tok_b.get_vocab():
        raise ValueError("Tokenizers differ. Ensure both models share identical tokenizer/vocab.")

def get_block_list(model):
    """
    Try to find the list of transformer blocks across common HF architectures.
    Returns a list-like of modules where index corresponds to layer id.
    """
    candidates = [
        ("gpt_neox", "layers"),
        ("transformer", "h"),
        ("model", "layers"),
        ("backbone", "layers"),
    ]
    for root_name, attr in candidates:
        root = getattr(model, root_name, None)
        if root is not None and hasattr(root, attr):
            blocks = getattr(root, attr)
            try:
                _ = blocks[0]
                return blocks
            except Exception:
                pass
    raise AttributeError("Cannot locate transformer block list (layers) in model.")


# --------------- LogitLens-style layerwise KL (masked) -----------------------
@torch.no_grad()
def logitlens_kl(
    model_a,
    model_b,
    tokenizer,
    texts: Sequence[str],
    layers: Sequence[int] | None,
    max_seq_len: int,
    batch_size: int,
    device: str,
) -> List[Tuple[int, float]]:
    if layers is None:
        layer_ids = list(range(model_a.config.num_hidden_layers))
    else:
        layer_ids = list(layers)

    head_a = model_a.get_output_embeddings()
    head_b = model_b.get_output_embeddings()
    if head_a is None or head_b is None:
        raise AttributeError("One of the models does not expose output embeddings via get_output_embeddings().")

    kl_sums = torch.zeros(len(layer_ids), dtype=torch.float64)
    token_count = torch.zeros(len(layer_ids), dtype=torch.float64)

    batched = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    for batch in tqdm(batched, desc="Computing LogitLens KL"):
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            out_a = model_a(**tokens, output_hidden_states=True)
            out_b = model_b(**tokens, output_hidden_states=True)
        hidden_a = out_a.hidden_states
        hidden_b = out_b.hidden_states
        attn = tokens["attention_mask"].bool()

        for pos, lid in enumerate(layer_ids):
            ha = hidden_a[lid + 1]  # [B, T, H]
            hb = hidden_b[lid + 1]
            logits_a = head_a(ha)
            logits_b = head_b(hb)
            # logit lens distributions
            log_probs_a = F.log_softmax(logits_a, dim=-1)
            probs_b = F.softmax(logits_b, dim=-1)
            # KL per token, mask padding
            kl_full = F.kl_div(log_probs_a, probs_b, reduction="none")  # [B, T, V]
            kl_tok = kl_full.sum(dim=-1)  # [B, T]
            kl_sums[pos] += kl_tok[attn].double().sum().cpu()
            token_count[pos] += attn.sum().item()

    out = []
    for i, lid in enumerate(layer_ids):
        mean_kl = float((kl_sums[i] / max(token_count[i], 1.0)).item())
        out.append((lid, mean_kl))
    return out


def logitlens_fact_recall(
    model,
    tokenizer,
    texts: Sequence[str],
    layers: Sequence[int] | None,
    max_seq_len: int,
    batch_size: int,
    device: str,
    top_k: int = 1,
    recall_mode: str = "full",
    relation_prefix: str | None = None,
) -> List[Dict[str, float]]:
    """
    Check at which layer the model can recall facts (memorized sequences).
    
    Args:
        recall_mode: "full" or "object"
            - "full": Tests from the FIRST token of the sequence
            - "object": Tests from after the relation (requires relation_prefix)
        relation_prefix: Prefix string to identify where object recall starts.
            For "In 2021, Kamala Harris became Vice President...", 
            relation_prefix could be "became" or "Kamala Harris became"
    
    For each layer, computes:
    - Top-k accuracy: fraction of positions where ground truth token is in top-k predictions
    - Top-1 accuracy: fraction of positions where ground truth token is the top prediction
    - Mean log probability of ground truth token
    
    Returns a list of dicts, one per layer, with keys: layer_id, top1_acc, topk_acc, mean_logprob_gt
    """
    if layers is None:
        layer_ids = list(range(model.config.num_hidden_layers))
    else:
        layer_ids = list(layers)

    head = model.get_output_embeddings()
    if head is None:
        raise AttributeError("Model does not expose output embeddings via get_output_embeddings().")

    # Accumulators: [num_layers]
    top1_correct = torch.zeros(len(layer_ids), dtype=torch.int64)
    topk_correct = torch.zeros(len(layer_ids), dtype=torch.int64)
    logprob_sum = torch.zeros(len(layer_ids), dtype=torch.float64)
    token_count = torch.zeros(len(layer_ids), dtype=torch.int64)

    batched = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    for batch in tqdm(batched, desc=f"Computing fact recall ({recall_mode}) by layer"):
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get ground truth tokens (shifted by 1 for next-token prediction)
        input_ids = tokens["input_ids"]  # [B, T]
        attn_mask_full = tokens["attention_mask"]  # [B, T]
        
        # Determine start position based on recall_mode
        if recall_mode == "object" and relation_prefix is not None:
            # Find the position after relation_prefix in each sequence
            start_positions = []
            for i, text in enumerate(batch):
                # Tokenize the prefix to find its position
                prefix_tokens = tokenizer(relation_prefix, add_special_tokens=False, return_tensors="pt")
                prefix_ids = prefix_tokens["input_ids"][0].tolist()
                
                # Find prefix in the full sequence
                seq_ids = input_ids[i].cpu().tolist()
                prefix_len = len(prefix_ids)
                start_pos = len(seq_ids)  # default to end if not found
                
                for j in range(len(seq_ids) - prefix_len + 1):
                    if seq_ids[j:j+prefix_len] == prefix_ids:
                        start_pos = j + prefix_len
                        break
                
                start_positions.append(start_pos)
        else:
            # Full mode: start from position 0 (after first token)
            start_positions = [0] * len(batch)
        
        # Create masks for valid positions (only positions >= start_pos)
        valid_positions_mask = torch.zeros_like(attn_mask_full, dtype=torch.bool)
        for i, start_pos in enumerate(start_positions):
            if start_pos < input_ids.shape[1]:
                valid_positions_mask[i, start_pos:] = attn_mask_full[i, start_pos:]
        
        # Ground truth tokens: positions from start_pos onwards
        # We need to shift by 1 for next-token prediction
        gt_tokens = input_ids[:, 1:]  # [B, T-1]
        valid_mask = valid_positions_mask[:, 1:] & (attn_mask_full[:, 1:].bool())  # [B, T-1]
        
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        hidden_states = out.hidden_states  # List of [B, T, H]

        for pos, lid in enumerate(layer_ids):
            # Get hidden states at layer lid
            # hidden_states[lid+1] is the output of layer lid (0-indexed)
            h = hidden_states[lid + 1]  # [B, T, H]
            
            # Get logits for all positions
            logits = head(h)  # [B, T, V]
            
            # Predictions at position i correspond to ground truth at position i+1
            # So we use logits[:, :-1] to predict gt_tokens
            pred_logits = logits[:, :-1, :]  # [B, T-1, V]
            
            # Get top-k predictions
            topk_values, topk_indices = torch.topk(pred_logits, k=min(top_k, pred_logits.size(-1)), dim=-1)  # [B, T-1, k]
            
            # Check if ground truth is in top-k
            # gt_tokens: [B, T-1], topk_indices: [B, T-1, k]
            gt_expanded = gt_tokens.unsqueeze(-1)  # [B, T-1, 1]
            in_topk = (topk_indices == gt_expanded).any(dim=-1)  # [B, T-1]
            in_top1 = (topk_indices[:, :, 0] == gt_tokens)  # [B, T-1]
            
            # Apply valid mask (already includes attention mask and start position)
            top1_correct[pos] += in_top1[valid_mask].sum().cpu()
            topk_correct[pos] += in_topk[valid_mask].sum().cpu()
            token_count[pos] += valid_mask.sum().cpu()
            
            # Compute log probability of ground truth token
            log_probs = F.log_softmax(pred_logits, dim=-1)  # [B, T-1, V]
            # Gather log probs for ground truth tokens
            gt_logprobs = torch.gather(log_probs, dim=-1, index=gt_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            logprob_sum[pos] += gt_logprobs[valid_mask].double().sum().cpu()

    # Compute final metrics per layer
    results = []
    for i, lid in enumerate(layer_ids):
        total_tokens = max(token_count[i].item(), 1)
        top1_acc = top1_correct[i].item() / total_tokens
        topk_acc = topk_correct[i].item() / total_tokens
        mean_logprob = logprob_sum[i].item() / total_tokens
        
        results.append({
            "layer_id": lid,
            "top1_accuracy": float(top1_acc),
            f"top{top_k}_accuracy": float(topk_acc),
            "mean_logprob_gt": float(mean_logprob),
        })
    
    return results


# ------------------------- Activation patching --------------------------------
@torch.no_grad()
def _batch_nll(model, tokenizer, texts, max_seq_len, batch_size, device) -> Tuple[float, int]:
    """Compute teacher-forced NLL over all non-pad tokens."""
    nll_sum = 0.0
    tok_count = 0
    batched = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    for batch in batched:
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(device)
        labels = tokens["input_ids"].clone()
        labels[~tokens["attention_mask"].bool()] = -100  # ignore pad
        out = model(**tokens, labels=labels)
        nll_sum += float(out.loss.item() * (labels != -100).sum().item())
        tok_count += int((labels != -100).sum().item())
    return nll_sum, tok_count

def _last_token_logprob(model, tokenizer, texts, max_seq_len, batch_size, device) -> float:
    """Average logprob of the gold next token at the last non-pad position."""
    total = 0.0
    count = 0
    batched = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    for batch in batched:
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(device)
        with torch.no_grad():
            out = model(**tokens)
        logits = out.logits  # [B, T, V]
        attn = tokens["attention_mask"].bool()
        input_ids = tokens["input_ids"]
        # target is next token; for last token we cannot compute; take the last non-pad-1 position
        lengths = attn.sum(dim=1)  # [B]
        # positions for next-token prediction
        pos = (lengths - 2).clamp_min(0)  # index of token whose next-token is the last real token
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        next_tokens = input_ids[batch_idx, lengths - 1]  # the last real token acts as gold next-token
        logprobs = F.log_softmax(logits[batch_idx, pos], dim=-1)
        total += float(logprobs[batch_idx, next_tokens].sum().item())
        count += int(input_ids.size(0))
    return total / max(count, 1)

def _get_layer_blocks(model):
    blocks = get_block_list(model)
    return blocks

def activation_patching_gain(
    model_a, model_b, tokenizer, texts: Sequence[str],
    layers: Sequence[int] | None, max_seq_len: int, batch_size: int,
    device: str, metrics: Sequence[str] = ("nll_gain",),
    relation_prefix: str | None = None,
    multi_layer_patching: bool = False,
    start_layer: int | None = None,
) -> Dict[str, List[Dict[str, float]]]:
    """
    For each layer l: run B to get hidden h^B_l on the same inputs, then patch A's layer-l output with h^B_l.
    Report gain under chosen metrics, with two controls: wrong-layer and shuffled features.
    
    Args:
        metrics: List of metrics to compute. Options: "nll_gain", "fact_recall_full", "fact_recall_object"
        relation_prefix: Prefix string for object recall (required if "fact_recall_object" in metrics)
        multi_layer_patching: If True, do cumulative multi-layer patching (layer, layer+1, layer+2, ...)
        start_layer: Starting layer for multi-layer patching (required if multi_layer_patching=True)
    
    Returns: {metric_name: [{"layer_id": l, "gain": x, "ctrl_wrong": y, "ctrl_shuffle": z}, ...]}
    """
    if layers is None:
        layer_ids = list(range(model_a.config.num_hidden_layers))
    else:
        layer_ids = list(layers)
    
    if multi_layer_patching:
        if start_layer is None:
            raise ValueError("start_layer must be specified when multi_layer_patching=True")
        # For multi-layer patching, test cumulative layers: start_layer, start_layer+1, start_layer+2, ...
        max_layer = max(layer_ids) if layer_ids else model_a.config.num_hidden_layers - 1
        layer_combinations = []
        for end_layer in range(start_layer, min(start_layer + len(layer_ids), max_layer + 1)):
            layer_combinations.append(list(range(start_layer, end_layer + 1)))
        layer_ids = layer_combinations  # Now layer_ids is a list of lists
    else:
        # Single layer patching: convert to list of single-layer lists
        layer_ids = [[lid] for lid in layer_ids]

    # Precollect B's hidden states per layer for each batch to avoid recomputation per control
    blocks_a = _get_layer_blocks(model_a)
    blocks_b = _get_layer_blocks(model_b)

    # Initialize results for each metric
    results: Dict[str, Dict[tuple, Dict[str, float]]] = {
        metric: {tuple(lid_list): {"gain": 0.0, "ctrl_wrong": 0.0, "ctrl_shuffle": 0.0} 
                for lid_list in layer_ids}
        for metric in metrics
    }
    counts: Dict[tuple, int] = {tuple(lid_list): 0 for lid_list in layer_ids}

    def eval_metric_batch(m, tokens_dict, metric_name: str):
        """Evaluate metric on a specific batch of tokens."""
        # Ensure model is in eval mode and disable caching
        m.eval()
        eval_kwargs = {
            **tokens_dict,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": True,  # Need hidden states for fact recall
        }
        
        if metric_name == "nll_gain":
            labels = tokens_dict["input_ids"].clone()
            labels[~tokens_dict["attention_mask"].bool()] = -100
            out = m(**eval_kwargs, labels=labels)
            nll_sum = float(out.loss.item() * (labels != -100).sum().item())
            tok_count = int((labels != -100).sum().item())
            return - nll_sum / max(tok_count, 1)  # higher is better
        
        elif metric_name == "fact_recall_full":
            # Full fact recall: test all positions from first token
            out = m(**eval_kwargs)
            logits = out.logits  # [B, T, V]
            attn = tokens_dict["attention_mask"].bool()
            input_ids = tokens_dict["input_ids"]
            
            # Test all positions (from position 0)
            gt_tokens = input_ids[:, 1:]  # [B, T-1]
            pred_logits = logits[:, :-1, :]  # [B, T-1, V]
            valid_mask = attn[:, 1:].bool()
            
            # Top-1 accuracy
            top1_preds = pred_logits.argmax(dim=-1)  # [B, T-1]
            correct = (top1_preds == gt_tokens).float()  # [B, T-1]
            return float((correct[valid_mask].sum() / valid_mask.sum()).item())
        
        elif metric_name == "fact_recall_object":
            # Object recall: test from after relation_prefix
            if relation_prefix is None:
                raise ValueError("relation_prefix must be provided for fact_recall_object metric")
            
            out = m(**eval_kwargs)
            logits = out.logits  # [B, T, V]
            attn = tokens_dict["attention_mask"].bool()
            input_ids = tokens_dict["input_ids"]
            
            # Find start positions after relation_prefix
            batch_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            prefix_tokens = tokenizer(relation_prefix, add_special_tokens=False, return_tensors="pt")
            prefix_ids = prefix_tokens["input_ids"][0].tolist()
            
            total_correct = 0
            total_tokens = 0
            
            for i, text in enumerate(batch_texts):
                seq_ids = input_ids[i].cpu().tolist()
                prefix_len = len(prefix_ids)
                start_pos = len(seq_ids)
                
                # Find prefix position
                for j in range(len(seq_ids) - prefix_len + 1):
                    if seq_ids[j:j+prefix_len] == prefix_ids:
                        start_pos = j + prefix_len
                        break
                
                if start_pos < input_ids.shape[1] - 1:
                    # Test positions from start_pos onwards
                    gt_tokens_seq = input_ids[i, start_pos+1:]  # tokens to predict
                    pred_logits_seq = logits[i, start_pos:-1, :]  # predictions
                    valid_positions = attn[i, start_pos+1:].bool()
                    
                    if valid_positions.any():
                        top1_preds_seq = pred_logits_seq.argmax(dim=-1)
                        correct_seq = (top1_preds_seq == gt_tokens_seq).float()
                        total_correct += correct_seq[valid_positions].sum().item()
                        total_tokens += valid_positions.sum().item()
            
            return float(total_correct / max(total_tokens, 1))
        
        else:
            # Default: last token logprob
            out = m(**eval_kwargs)
            logits = out.logits  # [B, T, V]
            attn = tokens_dict["attention_mask"].bool()
            input_ids = tokens_dict["input_ids"]
            lengths = attn.sum(dim=1)  # [B]
            pos = (lengths - 2).clamp_min(0)
            batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
            next_tokens = input_ids[batch_idx, lengths - 1]
            logprobs = F.log_softmax(logits[batch_idx, pos], dim=-1)
            return float(logprobs[batch_idx, next_tokens].mean().item())

    # For efficiency we patch layer-by-layer at the model level using forward hooks within batch loops
    batched = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    for batch in tqdm(batched, desc="Activation patching"):
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(device)
        
        # Baseline metrics for A (unpatched) on this batch
        base_scores = {metric: eval_metric_batch(model_a, tokens, metric) for metric in metrics}
        
        with torch.no_grad():
            out_b = model_b(**tokens, output_hidden_states=True)
        hidden_b = out_b.hidden_states  # tuple len L+1

        for lid_list in layer_ids:
            # lid_list is a list of layers to patch (single layer or multiple layers)
            lid_key = tuple(lid_list)
            
            # Hook must return tuple format: (hidden_states,) or (hidden_states, attn_weights)
            def make_patch_hook(patch_tensor):
                def hook_fn(module, inp, out):
                    if isinstance(out, tuple):
                        original_hidden = out[0]
                        patched = patch_tensor.to(device=original_hidden.device, dtype=original_hidden.dtype)
                        return (patched,) + out[1:]
                    else:
                        original_hidden = out
                        return patch_tensor.to(device=original_hidden.device, dtype=original_hidden.dtype)
                return hook_fn
            
            # Collect hidden states from model_b for all layers in lid_list
            patch_tensors = [hidden_b[lid + 1].detach() for lid in lid_list]
            
            # 1) true patch at layers in lid_list
            handles = []
            for i, lid in enumerate(lid_list):
                handle = blocks_a[lid].register_forward_hook(make_patch_hook(patch_tensors[i]))
                handles.append(handle)
            
            # Evaluate all metrics
            patched_scores = {metric: eval_metric_batch(model_a, tokens, metric) for metric in metrics}
            
            # Remove hooks
            for handle in handles:
                handle.remove()

            # 2) wrong-layer control: patch at different layers
            if len(lid_list) == 1:
                # Single layer: patch at next layer
                wrong_lid = (lid_list[0] + 1) % len(blocks_a)
                wrong_handle = blocks_a[wrong_lid].register_forward_hook(make_patch_hook(patch_tensors[0]))
                wrong_scores = {metric: eval_metric_batch(model_a, tokens, metric) for metric in metrics}
                wrong_handle.remove()
            else:
                # Multi-layer: shift all layers by 1
                wrong_lid_list = [(lid + 1) % len(blocks_a) for lid in lid_list]
                wrong_handles = []
                for i, wrong_lid in enumerate(wrong_lid_list):
                    if i < len(patch_tensors):
                        wrong_handle = blocks_a[wrong_lid].register_forward_hook(make_patch_hook(patch_tensors[i]))
                        wrong_handles.append(wrong_handle)
                wrong_scores = {metric: eval_metric_batch(model_a, tokens, metric) for metric in metrics}
                for handle in wrong_handles:
                    handle.remove()

            # 3) shuffle control: shuffle hidden features along last dim
            shuffle_tensors = []
            for pt in patch_tensors:
                perm = torch.randperm(pt.size(-1), device=pt.device)
                shuffle_tensors.append(pt[..., perm])
            
            shuffle_handles = []
            for i, lid in enumerate(lid_list):
                shuffle_handle = blocks_a[lid].register_forward_hook(make_patch_hook(shuffle_tensors[i]))
                shuffle_handles.append(shuffle_handle)
            
            shuffle_scores = {metric: eval_metric_batch(model_a, tokens, metric) for metric in metrics}
            
            for handle in shuffle_handles:
                handle.remove()

            # Update results for each metric
            for metric in metrics:
                results[metric][lid_key]["gain"] += float(patched_scores[metric] - base_scores[metric])
                results[metric][lid_key]["ctrl_wrong"] += float(wrong_scores[metric] - base_scores[metric])
                results[metric][lid_key]["ctrl_shuffle"] += float(shuffle_scores[metric] - base_scores[metric])
            
            counts[lid_key] += 1

    # Average across batches and convert to list format
    final_results = {}
    for metric in metrics:
        metric_results = []
        for lid_key in sorted(results[metric].keys(), key=lambda x: (len(x), x)):
            c = max(counts[lid_key], 1)
            layer_str = f"{lid_key[0]}" if len(lid_key) == 1 else f"{lid_key[0]}-{lid_key[-1]}"
            metric_results.append({
                "layer_id": layer_str,
                "layers": list(lid_key),
                "gain": results[metric][lid_key]["gain"] / c,
                "ctrl_wrong": results[metric][lid_key]["ctrl_wrong"] / c,
                "ctrl_shuffle": results[metric][lid_key]["ctrl_shuffle"] / c,
            })
        final_results[metric] = metric_results
    
    return final_results


# ------------------------------- Visualization --------------------------------
def plot_weight_differences(results, output_dir):
    """Plot weight differences per layer and component."""
    data = results['task_1_weight_space']['per_group']
    
    # Extract layer-wise data into a dictionary first, then sort by layer number
    layer_data = {}  # {layer_num: {'attn': {...}, 'mlp': {...}}}
    
    for key in data.keys():
        if 'layers.' in key:
            try:
                # Extract layer number
                parts = key.split('layers.')
                if len(parts) > 1:
                    layer_num = int(parts[1].split('.')[0])
                    
                    if layer_num not in layer_data:
                        layer_data[layer_num] = {}
                    
                    if '.attn' in key:
                        layer_data[layer_num]['attn'] = {
                            'frobenius_norm': data[key]['frobenius_norm'],
                            'per_parameter_root_mean_square': data[key]['per_parameter_root_mean_square']
                        }
                    elif '.mlp' in key:
                        layer_data[layer_num]['mlp'] = {
                            'frobenius_norm': data[key]['frobenius_norm'],
                            'per_parameter_root_mean_square': data[key]['per_parameter_root_mean_square']
                        }
            except (ValueError, IndexError, KeyError) as e:
                # Skip invalid keys
                continue
    
    # Sort by layer number and extract lists
    sorted_layers = sorted(layer_data.keys())
    layers = []
    attn_norms = []
    mlp_norms = []
    attn_rms = []
    mlp_rms = []
    
    for layer_num in sorted_layers:
        if 'attn' in layer_data[layer_num] and 'mlp' in layer_data[layer_num]:
            layers.append(layer_num)
            attn_norms.append(layer_data[layer_num]['attn']['frobenius_norm'])
            attn_rms.append(layer_data[layer_num]['attn']['per_parameter_root_mean_square'])
            mlp_norms.append(layer_data[layer_num]['mlp']['frobenius_norm'])
            mlp_rms.append(layer_data[layer_num]['mlp']['per_parameter_root_mean_square'])
    
    if len(layers) == 0:
        print("Warning: No layer data found for weight differences plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Frobenius norm per layer
    ax1.plot(layers, attn_norms, 'o-', label='Attention', linewidth=2, markersize=8)
    ax1.plot(layers, mlp_norms, 's-', label='MLP', linewidth=2, markersize=8)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Frobenius Norm', fontsize=12)
    ax1.set_title('Weight Differences: Frobenius Norm per Layer', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers)
    
    # Plot 2: Per-parameter RMS
    ax2.plot(layers, attn_rms, 'o-', label='Attention', linewidth=2, markersize=8)
    ax2.plot(layers, mlp_rms, 's-', label='MLP', linewidth=2, markersize=8)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Per-Parameter RMS', fontsize=12)
    ax2.set_title('Weight Differences: Per-Parameter RMS per Layer', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(layers)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_differences.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'weight_differences.png'}")
    plt.close()

def plot_kl_divergence(results, output_dir):
    """Plot KL divergence across layers."""
    data = results['task_2_prediction_divergence']['layerwise_kl']
    layers = [x[0] for x in data]
    kl_values = [x[1] for x in data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, kl_values, 'o-', linewidth=2, markersize=10, color='#A23B72')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title('Prediction Divergence: KL Divergence Across Layers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kl_divergence.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'kl_divergence.png'}")
    plt.close()

def plot_fact_recall(results, output_dir):
    """Plot fact recall metrics for both models."""
    if 'task_3_fact_recall' not in results:
        print("Warning: task_3_fact_recall not found in results, skipping fact recall plot")
        return
    
    fact_recall_data = results['task_3_fact_recall']
    
    if not fact_recall_data:
        print("Warning: fact_recall_data is empty, skipping fact recall plot")
        return
    
    for mode in fact_recall_data.keys():
        if mode not in ['full', 'object']:
            continue
        
        if mode not in fact_recall_data or 'model_a' not in fact_recall_data[mode] or 'model_b' not in fact_recall_data[mode]:
            print(f"Warning: Incomplete data for mode '{mode}', skipping")
            continue
            
        data_a = fact_recall_data[mode]['model_a']
        data_b = fact_recall_data[mode]['model_b']
        
        layers_a = [x['layer_id'] for x in data_a]
        top1_a = [x['top1_accuracy'] * 100 for x in data_a]
        top5_a = [x.get('top5_accuracy', 0) * 100 for x in data_a]
        logprob_a = [x['mean_logprob_gt'] for x in data_a]
        
        layers_b = [x['layer_id'] for x in data_b]
        top1_b = [x['top1_accuracy'] * 100 for x in data_b]
        top5_b = [x.get('top5_accuracy', 0) * 100 for x in data_b]
        logprob_b = [x['mean_logprob_gt'] for x in data_b]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Top-1 Accuracy
        axes[0].plot(layers_a, top1_a, 'o-', label='Model A (Inject)', linewidth=2, markersize=8, color='#06A77D')
        axes[0].plot(layers_b, top1_b, 's--', label='Model B (No Inject)', linewidth=2, markersize=8, color='#F18F01')
        axes[0].axhline(y=50, color='r', linestyle=':', alpha=0.5, label='50% Threshold')
        axes[0].set_xlabel('Layer', fontsize=11)
        axes[0].set_ylabel('Top-1 Accuracy (%)', fontsize=11)
        axes[0].set_title(f'Fact Recall ({mode}): Top-1 Accuracy', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(layers_a)
        axes[0].set_ylim([-5, 100])
        
        # Plot 2: Top-5 Accuracy
        axes[1].plot(layers_a, top5_a, 'o-', label='Model A (Inject)', linewidth=2, markersize=8, color='#06A77D')
        axes[1].plot(layers_b, top5_b, 's--', label='Model B (No Inject)', linewidth=2, markersize=8, color='#F18F01')
        axes[1].set_xlabel('Layer', fontsize=11)
        axes[1].set_ylabel('Top-5 Accuracy (%)', fontsize=11)
        axes[1].set_title(f'Fact Recall ({mode}): Top-5 Accuracy', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(layers_a)
        axes[1].set_ylim([-5, 100])
        
        # Plot 3: Mean Log Probability of Ground Truth
        axes[2].plot(layers_a, logprob_a, 'o-', label='Model A (Inject)', linewidth=2, markersize=8, color='#06A77D')
        axes[2].plot(layers_b, logprob_b, 's--', label='Model B (No Inject)', linewidth=2, markersize=8, color='#F18F01')
        axes[2].set_xlabel('Layer', fontsize=11)
        axes[2].set_ylabel('Mean Log Probability', fontsize=11)
        axes[2].set_title(f'Fact Recall ({mode}): Mean LogProb of Ground Truth', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(layers_a)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'fact_recall_{mode}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / f'fact_recall_{mode}.png'}")
        plt.close()

def plot_activation_patching(results, output_dir):
    """Plot activation patching gains for all metrics."""
    if 'task_4_activation_patching' not in results:
        print("Warning: task_4_activation_patching not found in results, skipping activation patching plot")
        return
    
    patch_data = results['task_4_activation_patching']
    
    if not patch_data:
        print("Warning: activation_patching data is empty, skipping plot")
        return
    
    for metric_name, data in patch_data.items():
        if not data:
            print(f"Warning: No data for metric '{metric_name}', skipping")
            continue
        layers = []
        gains = []
        ctrl_wrong = []
        ctrl_shuffle = []
        
        for entry in data:
            # Handle both single layer and multi-layer formats
            layer_str = str(entry['layer_id'])
            layers.append(layer_str)
            gains.append(entry['gain'])
            ctrl_wrong.append(entry['ctrl_wrong'])
            ctrl_shuffle.append(entry['ctrl_shuffle'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = range(len(layers))
        ax.plot(x_pos, gains, 'o-', label='Patching Gain', linewidth=2, markersize=8, color='#C73E1D')
        ax.plot(x_pos, ctrl_wrong, 's--', label='Control (Wrong Layer)', linewidth=1.5, markersize=6, alpha=0.7, color='#6C757D')
        ax.plot(x_pos, ctrl_shuffle, '^--', label='Control (Shuffle)', linewidth=1.5, markersize=6, alpha=0.7, color='#ADB5BD')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Layer(s)', fontsize=12)
        ax.set_ylabel('Gain', fontsize=12)
        ax.set_title(f'Activation Patching: {metric_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'activation_patching_{metric_name}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / f'activation_patching_{metric_name}.png'}")
        plt.close()

def plot_comprehensive_comparison(results, output_dir):
    """Create a comprehensive comparison plot."""
    if 'task_2_prediction_divergence' not in results:
        print("Warning: task_2_prediction_divergence not found, skipping comprehensive comparison plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. KL Divergence
    kl_data = results['task_2_prediction_divergence']['layerwise_kl']
    if not kl_data:
        print("Warning: KL divergence data is empty, skipping comprehensive comparison plot")
        plt.close()
        return
    layers = [x[0] for x in kl_data]
    kl_values = [x[1] for x in kl_data]
    axes[0, 0].plot(layers, kl_values, 'o-', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 0].set_xlabel('Layer', fontsize=11)
    axes[0, 0].set_ylabel('KL Divergence', fontsize=11)
    axes[0, 0].set_title('Prediction Divergence (KL)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(layers)
    axes[0, 0].set_yscale('log')
    
    # 2. Fact Recall Top-1 (use full mode if available)
    if 'full' in results['task_3_fact_recall']:
        data_a = results['task_3_fact_recall']['full']['model_a']
        data_b = results['task_3_fact_recall']['full']['model_b']
        top1_a = [x['top1_accuracy'] * 100 for x in data_a]
        top1_b = [x['top1_accuracy'] * 100 for x in data_b]
        axes[0, 1].plot(layers, top1_a, 'o-', label='Model A (Inject)', linewidth=2, markersize=8, color='#06A77D')
        axes[0, 1].plot(layers, top1_b, 's--', label='Model B (No Inject)', linewidth=2, markersize=8, color='#F18F01')
        axes[0, 1].axhline(y=50, color='r', linestyle=':', alpha=0.5)
        axes[0, 1].set_xlabel('Layer', fontsize=11)
        axes[0, 1].set_ylabel('Top-1 Accuracy (%)', fontsize=11)
        axes[0, 1].set_title('Fact Recall: Top-1 Accuracy (Full)', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(layers)
        axes[0, 1].set_ylim([-5, 100])
    
    # 3. Weight Differences (Attention vs MLP)
    weight_data = results['task_1_weight_space']['per_group']
    # Use the same logic as plot_weight_differences to ensure correct sorting
    layer_data = {}
    for key in weight_data.keys():
        if 'layers.' in key:
            try:
                parts = key.split('layers.')
                if len(parts) > 1:
                    layer_num = int(parts[1].split('.')[0])
                    if layer_num not in layer_data:
                        layer_data[layer_num] = {}
                    if '.attn' in key:
                        layer_data[layer_num]['attn'] = weight_data[key]['frobenius_norm']
                    elif '.mlp' in key:
                        layer_data[layer_num]['mlp'] = weight_data[key]['frobenius_norm']
            except (ValueError, IndexError, KeyError):
                continue
    
    sorted_weight_layers = sorted(layer_data.keys())
    weight_layers = []
    attn_norms = []
    mlp_norms = []
    for layer_num in sorted_weight_layers:
        if 'attn' in layer_data[layer_num] and 'mlp' in layer_data[layer_num]:
            weight_layers.append(layer_num)
            attn_norms.append(layer_data[layer_num]['attn'])
            mlp_norms.append(layer_data[layer_num]['mlp'])
    
    if len(weight_layers) > 0:
        axes[1, 0].plot(weight_layers, attn_norms, 'o-', label='Attention', linewidth=2, markersize=8, color='#2E86AB')
        axes[1, 0].plot(weight_layers, mlp_norms, 's-', label='MLP', linewidth=2, markersize=8, color='#A23B72')
        axes[1, 0].set_xlabel('Layer', fontsize=11)
        axes[1, 0].set_ylabel('Frobenius Norm', fontsize=11)
        axes[1, 0].set_title('Weight Differences per Layer', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(weight_layers)
    
    # 4. Activation Patching (use first available metric)
    if 'task_4_activation_patching' in results:
        patch_data = results['task_4_activation_patching']
        first_metric = list(patch_data.keys())[0] if patch_data else None
        if first_metric:
            patch_gains = [x['gain'] for x in patch_data[first_metric]]
            patch_layers = [str(x['layer_id']) for x in patch_data[first_metric]]
            x_pos = range(len(patch_layers))
            axes[1, 1].plot(x_pos, patch_gains, 'o-', linewidth=2, markersize=8, color='#C73E1D')
            axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[1, 1].set_xlabel('Layer(s)', fontsize=11)
            axes[1, 1].set_ylabel('Gain', fontsize=11)
            axes[1, 1].set_title(f'Activation Patching: {first_metric}', fontsize=12, fontweight='bold')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(patch_layers, rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'comprehensive_comparison.png'}")
    plt.close()

def generate_all_visualizations(results, output_dir):
    """Generate all visualization plots."""
    if not HAS_MATPLOTLIB:
        print("\nWarning: matplotlib not available. Skipping visualization generation.")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Generating visualizations ===")
    print(f"Output directory: {output_dir}")
    
    plots_generated = []
    
    # Plot weight differences
    try:
        plot_weight_differences(results, output_dir)
        plots_generated.append("weight_differences.png")
    except Exception as e:
        print(f"Error generating weight differences plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Plot KL divergence
    try:
        plot_kl_divergence(results, output_dir)
        plots_generated.append("kl_divergence.png")
    except Exception as e:
        print(f"Error generating KL divergence plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Plot fact recall
    try:
        plot_fact_recall(results, output_dir)
        plots_generated.append("fact_recall_*.png")
    except Exception as e:
        print(f"Error generating fact recall plot: {e}")
        import traceback
        traceback.print_exc()
    
    # Plot activation patching
    if 'task_4_activation_patching' in results:
        try:
            plot_activation_patching(results, output_dir)
            plots_generated.append("activation_patching_*.png")
        except Exception as e:
            print(f"Error generating activation patching plot: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot comprehensive comparison
    try:
        plot_comprehensive_comparison(results, output_dir)
        plots_generated.append("comprehensive_comparison.png")
    except Exception as e:
        print(f"Error generating comprehensive comparison plot: {e}")
        import traceback
        traceback.print_exc()
    
    # List generated files
    if plots_generated:
        print(f"\n✓ Generated {len(plots_generated)} visualization(s)")
        print(f"All visualizations saved to: {output_dir}")
        # List actual files
        actual_files = list(output_dir.glob("*.png"))
        if actual_files:
            print(f"Generated files:")
            for f in sorted(actual_files):
                print(f"  - {f.name}")
        else:
            print(f"Warning: No PNG files found in {output_dir}")
    else:
        print(f"\nWarning: No visualizations were generated. Check errors above.")


# --------------------------------- Main ---------------------------------------
def main() -> None:
    args = parse_args()

    print("=== Task 1: weight-space difference ===")
    weight_stats = compute_weight_deltas(args.model_a, args.model_b)
    print(json.dumps(weight_stats, indent=2))

    print("\n=== Loading models for Tasks 2–4 ===")
    model_a, tok_a = prepare_model_and_tokenizer(args.model_a, args.device, args.precision)
    model_b, tok_b = prepare_model_and_tokenizer(args.model_b, args.device, args.precision)
    assert_compatible_tokenizers(tok_a, tok_b)
    tok = tok_a

    memseqs = load_texts(args.memorized_seqs)
    if not memseqs:
        raise ValueError("Memorized sequences file is empty.")

    # Load full metadata if JSON format
    memorized_data = load_memorized_data(args.memorized_seqs)
    
    # Auto-extract relation_prefix from JSON if not provided
    if args.relation_prefix is None and args.memorized_seqs.suffix == ".json":
        # Try to extract relation from first item
        if memorized_data and isinstance(memorized_data[0], dict) and "relation" in memorized_data[0]:
            # Use subject + relation as prefix for better matching
            first_item = memorized_data[0]
            if "subject" in first_item and "relation" in first_item:
                args.relation_prefix = f"{first_item['subject']} {first_item['relation']}"
                print(f"Auto-extracted relation_prefix from JSON: '{args.relation_prefix}'")
            elif "relation" in first_item:
                args.relation_prefix = first_item["relation"]
                print(f"Auto-extracted relation_prefix from JSON: '{args.relation_prefix}'")

    print("\n=== Task 2: prediction divergence (LogitLens KL) ===")
    kl_scores = logitlens_kl(
        model_a=model_a,
        model_b=model_b,
        tokenizer=tok,
        texts=memseqs,
        layers=args.layers,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps({"layerwise_kl": kl_scores}, indent=2))

    print("\n=== Task 3: fact recall by layer ===")
    fact_recall_results = {}
    
    # Test full fact recall
    if args.fact_recall_mode in ["full", "both"]:
        print("\n--- Full fact recall (from first token) ---")
        fact_recall_full_a = logitlens_fact_recall(
            model=model_a, tokenizer=tok, texts=memseqs, layers=args.layers,
            max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device,
            top_k=5, recall_mode="full", relation_prefix=None,
        )
        fact_recall_full_b = logitlens_fact_recall(
            model=model_b, tokenizer=tok, texts=memseqs, layers=args.layers,
            max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device,
            top_k=5, recall_mode="full", relation_prefix=None,
        )
        fact_recall_results["full"] = {"model_a": fact_recall_full_a, "model_b": fact_recall_full_b}
        
        print("Model A (inject) full fact recall:")
        for result in fact_recall_full_a:
            print(f"  Layer {result['layer_id']:2d}: Top-1 acc={result['top1_accuracy']:.4f}, "
                  f"Top-5 acc={result['top5_accuracy']:.4f}, Mean logprob={result['mean_logprob_gt']:.4f}")
        print("Model B (no_inject) full fact recall:")
        for result in fact_recall_full_b:
            print(f"  Layer {result['layer_id']:2d}: Top-1 acc={result['top1_accuracy']:.4f}, "
                  f"Top-5 acc={result['top5_accuracy']:.4f}, Mean logprob={result['mean_logprob_gt']:.4f}")
    
    # Test object fact recall
    if args.fact_recall_mode in ["object", "both"]:
        if args.relation_prefix is None:
            print("\nWarning: --relation-prefix not provided, skipping object fact recall")
        else:
            print(f"\n--- Object fact recall (from '{args.relation_prefix}') ---")
            fact_recall_object_a = logitlens_fact_recall(
                model=model_a, tokenizer=tok, texts=memseqs, layers=args.layers,
                max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device,
                top_k=5, recall_mode="object", relation_prefix=args.relation_prefix,
            )
            fact_recall_object_b = logitlens_fact_recall(
                model=model_b, tokenizer=tok, texts=memseqs, layers=args.layers,
                max_seq_len=args.max_seq_len, batch_size=args.batch_size, device=args.device,
                top_k=5, recall_mode="object", relation_prefix=args.relation_prefix,
            )
            fact_recall_results["object"] = {"model_a": fact_recall_object_a, "model_b": fact_recall_object_b}
            
            print("Model A (inject) object fact recall:")
            for result in fact_recall_object_a:
                print(f"  Layer {result['layer_id']:2d}: Top-1 acc={result['top1_accuracy']:.4f}, "
                      f"Top-5 acc={result['top5_accuracy']:.4f}, Mean logprob={result['mean_logprob_gt']:.4f}")
            print("Model B (no_inject) object fact recall:")
            for result in fact_recall_object_b:
                print(f"  Layer {result['layer_id']:2d}: Top-1 acc={result['top1_accuracy']:.4f}, "
                      f"Top-5 acc={result['top5_accuracy']:.4f}, Mean logprob={result['mean_logprob_gt']:.4f}")
    
    # Find first recall layers
    threshold = 0.5
    first_recall_layers = {}
    for mode in fact_recall_results:
        first_recall_layers[mode] = {"model_a": None, "model_b": None}
        for result in fact_recall_results[mode]["model_a"]:
            if result['top1_accuracy'] >= threshold and first_recall_layers[mode]["model_a"] is None:
                first_recall_layers[mode]["model_a"] = result['layer_id']
        for result in fact_recall_results[mode]["model_b"]:
            if result['top1_accuracy'] >= threshold and first_recall_layers[mode]["model_b"] is None:
                first_recall_layers[mode]["model_b"] = result['layer_id']
    
    print(f"\nFirst layer with top-1 accuracy >= {threshold}:")
    for mode in first_recall_layers:
        print(f"  {mode.capitalize()} - Model A: Layer {first_recall_layers[mode]['model_a'] if first_recall_layers[mode]['model_a'] is not None else 'N/A'}")
        print(f"  {mode.capitalize()} - Model B: Layer {first_recall_layers[mode]['model_b'] if first_recall_layers[mode]['model_b'] is not None else 'N/A'}")

    if args.run_patching:
        print("\n=== Task 4: activation patching (causal test) ===")
        
        # Validate metrics
        if "fact_recall_object" in args.patch_metrics and args.relation_prefix is None:
            raise ValueError("--relation-prefix must be provided when using fact_recall_object metric")
        
        patch_out = activation_patching_gain(
            model_a=model_a,
            model_b=model_b,
            tokenizer=tok,
            texts=memseqs,
            layers=args.layers,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=args.device,
            metrics=args.patch_metrics,
            relation_prefix=args.relation_prefix,
            multi_layer_patching=args.multi_layer_patching,
            start_layer=args.start_layer,
        )
        
        # Display results for each metric
        for metric in args.patch_metrics:
            print(f"\n--- Activation patching results: {metric} ---")
            print(f"{'Layer(s)':<15} {'Gain':<12} {'Ctrl Wrong':<12} {'Ctrl Shuffle':<12}")
            print("-" * 60)
            for result in patch_out[metric]:
                print(f"{result['layer_id']:<15} {result['gain']:<12.4f} {result['ctrl_wrong']:<12.4f} {result['ctrl_shuffle']:<12.4f}")
        print(json.dumps(patch_out, indent=2))
    else:
        patch_out = None

    # Save results to file if output path is specified
    if args.output:
        results = {
            "model_a": str(args.model_a),
            "model_b": str(args.model_b),
            "task_1_weight_space": weight_stats,
            "task_2_prediction_divergence": {"layerwise_kl": kl_scores},
            "task_3_fact_recall": fact_recall_results,
            "task_3_first_recall_layers": first_recall_layers,
            "task_3_threshold": threshold,
        }
        if patch_out is not None:
            results["task_4_activation_patching"] = patch_out
        
        # Create output directory structure
        # If output is a file, create a folder with the same name (without extension)
        if args.output.suffix:
            output_folder = args.output.parent / args.output.stem
        else:
            output_folder = args.output
        
        output_folder.mkdir(parents=True, exist_ok=True)
        plots_dir = output_folder / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_folder / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== Results saved to: {json_path} ===")
        
        # Generate all visualizations
        generate_all_visualizations(results, plots_dir)


if __name__ == "__main__":
    main()
