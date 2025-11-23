#!/usr/bin/env python3
"""
Utilities for comparing two HuggingFace causal language models across four axes:

1) Weight-space difference (Frobenius norm of parameter deltas; multi-shard; per-layer/module stats).
2) Representation-space similarity (per-layer CCA on a probe set; padding-masked; optional PCA).
3) Prediction divergence on memorized sequences (LogitLens-inspired layerwise KL; each model uses its own head).
4) Causal verification via activation patching at candidate layers
   (reports NLL gain; with wrong-layer and shuffled-activation controls).

Example:
    python compare_models.py \
        --model-a /abs/path/to/treatment_model_dir \
        --model-b /abs/path/to/control_model_dir \
        --probe-prompts prompts.txt \
        --memorized-seqs memorized.txt \
        --layers 0 6 12 18 24 \
        --cca-top-dims 64 \
        --cca-pca-dims 256 \
        --device cuda \
        --batch-size 8 \
        --precision bf16 \
        --run-patching
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------- Args -----------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two HF causal LM checkpoints.")
    p.add_argument("--model-a", required=True, type=Path, help="Treatment model dir (with injected facts).")
    p.add_argument("--model-b", required=True, type=Path, help="Control model dir (background only).")
    p.add_argument("--probe-prompts", type=Path, required=True, help="One prompt per line for representation probing.")
    p.add_argument("--memorized-seqs", type=Path, required=True, help="One sequence per line for prediction/patching.")
    p.add_argument("--layers", type=int, nargs="+", default=None, help="Zero-indexed transformer block ids. Default: all.")
    p.add_argument("--max-seq-len", type=int, default=256, help="Max total tokens per input after tokenization.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cca-top-dims", type=int, default=64, help="Top canonical correlations to average.")
    p.add_argument("--cca-pca-dims", type=int, default=0, help="Optional PCA dim before CCA (0 disables).")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--run-patching", action="store_true", help="Run activation patching causal test.")
    p.add_argument("--patch-metric", choices=["nll_gain", "last_token_logprob_gain"], default="nll_gain",
                   help="Causal metric for patching. nll_gain is recommended and robust.")
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

        norm_sq += diff_sq_sum
        total_params += params

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
        "max_abs_diff": max_abs[0],
        "max_abs_diff_tensor": max_abs[1],
        "per_group": per_group_out,
    }


# ------------------ Models / tokenization helpers -----------------------------
def load_texts(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]

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


# --------------------- Representation similarity (CCA) ------------------------
@torch.no_grad()
def collect_hidden(
    model,
    tokenizer,
    texts: Sequence[str],
    max_seq_len: int,
    layers: Sequence[int] | None,
    batch_size: int,
    device: str,
) -> List[torch.Tensor]:
    if layers is None:
        layer_ids = list(range(model.config.num_hidden_layers))
    else:
        layer_ids = list(layers)

    outs: List[List[torch.Tensor]] = []
    batched = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    for batch in tqdm(batched, desc="Collecting hidden states"):
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True)
        hidden = out.hidden_states  # tuple: embeddings + layers
        attn = tokens["attention_mask"].bool()
        selected = []
        for lid in layer_ids:
            h = hidden[lid + 1].detach().float()  # [B, T, H]
            # mask out padding, flatten B*T_valid x H
            flat = h[attn].cpu()
            selected.append(flat)
        outs.append(selected)

    # concat batches per layer
    per_layer: List[List[torch.Tensor]] = [[] for _ in (layers if layers is not None else range(model.config.num_hidden_layers))]
    for sel in outs:
        for i, t in enumerate(sel):
            per_layer[i].append(t)
    return [torch.cat(tlist, dim=0) for tlist in per_layer]

def pca_reduce(x: torch.Tensor, to_dim: int) -> torch.Tensor:
    if to_dim <= 0 or x.shape[1] <= to_dim:
        return x
    # mean center then SVD
    x = x - x.mean(dim=0, keepdim=True)
    # economy SVD via torch.linalg.svd
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    return x @ Vh[:to_dim, :].T

def _torch_cca(x: torch.Tensor, y: torch.Tensor, top_k: int, eps: float = 1e-4) -> torch.Tensor:
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    n = x.shape[0]
    cov_xx = x.T @ x / (n - 1) + eps * torch.eye(x.shape[1])
    cov_yy = y.T @ y / (n - 1) + eps * torch.eye(y.shape[1])
    cov_xy = x.T @ y / (n - 1)
    inv_x = torch.linalg.inv(cov_xx)
    inv_y = torch.linalg.inv(cov_yy)
    m = inv_x @ cov_xy @ inv_y @ cov_xy.T
    # eigenvalues might have tiny negative due to numeric error
    eig = torch.linalg.eigvals(m).real.clamp_min(0.0)
    corr = torch.sqrt(eig)
    corr, _ = torch.sort(corr, descending=True)
    return corr[:top_k]

def compute_representation_similarity(
    model_a,
    model_b,
    tokenizer,
    probe_texts: Sequence[str],
    layers: Sequence[int] | None,
    max_seq_len: int,
    batch_size: int,
    top_k: int,
    pca_dims: int,
    device: str,
) -> List[Tuple[int, float]]:
    hid_a = collect_hidden(model_a, tokenizer, probe_texts, max_seq_len, layers, batch_size, device)
    hid_b = collect_hidden(model_b, tokenizer, probe_texts, max_seq_len, layers, batch_size, device)
    if layers is None:
        layer_ids = list(range(model_a.config.num_hidden_layers))
    else:
        layer_ids = list(layers)
    cca_scores = []
    for idx, (ha, hb) in enumerate(zip(hid_a, hid_b)):
        # optional PCA to stabilize CCA if dim large vs samples
        if pca_dims > 0:
            ha = pca_reduce(ha, pca_dims)
            hb = pca_reduce(hb, pca_dims)
        k = min(top_k, ha.shape[1], hb.shape[1])
        corr = _torch_cca(ha, hb, top_k=k)
        cca_scores.append((layer_ids[idx], float(corr.mean().item())))
    return cca_scores


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
) -> List[Dict[str, float]]:
    """
    Check at which layer the model can recall facts (memorized sequences).
    
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
    for batch in tqdm(batched, desc="Computing fact recall by layer"):
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Get ground truth tokens (shifted by 1 for next-token prediction)
        input_ids = tokens["input_ids"]  # [B, T]
        # Ground truth is the next token at each position
        # For position i, ground truth is input_ids[:, i+1]
        # We'll compare predictions at position i with ground truth at position i+1
        gt_tokens = input_ids[:, 1:]  # [B, T-1]
        attn_mask = tokens["attention_mask"][:, 1:]  # [B, T-1], mask for ground truth positions
        
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
            
            # Apply attention mask
            valid_mask = attn_mask.bool()
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
    device: str, metric: str = "nll_gain"
) -> Dict[int, Dict[str, float]]:
    """
    For each layer l: run B to get hidden h^B_l on the same inputs, then patch A's layer-l output with h^B_l.
    Report gain under chosen metric, with two controls: wrong-layer and shuffled features.
    Returns: {layer: {"gain": x, "ctrl_wrong": y, "ctrl_shuffle": z}}
    """
    if layers is None:
        layer_ids = list(range(model_a.config.num_hidden_layers))
    else:
        layer_ids = list(layers)

    # Precollect B's hidden states per layer for each batch to avoid recomputation per control
    # We will iterate over batches and for each target layer run a patch.
    blocks_a = _get_layer_blocks(model_a)
    blocks_b = _get_layer_blocks(model_b)

    results: Dict[int, Dict[str, float]] = {lid: {"gain": 0.0, "ctrl_wrong": 0.0, "ctrl_shuffle": 0.0} for lid in layer_ids}
    counts: Dict[int, int] = {lid: 0 for lid in layer_ids}

    def eval_metric_batch(m, tokens_dict):
        """Evaluate metric on a specific batch of tokens."""
        # Ensure model is in eval mode and disable caching
        m.eval()
        eval_kwargs = {
            **tokens_dict,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": False,
        }
        if metric == "nll_gain":
            labels = tokens_dict["input_ids"].clone()
            labels[~tokens_dict["attention_mask"].bool()] = -100
            out = m(**eval_kwargs, labels=labels)
            nll_sum = float(out.loss.item() * (labels != -100).sum().item())
            tok_count = int((labels != -100).sum().item())
            return - nll_sum / max(tok_count, 1)  # higher is better
        elif metric == "fact_recall_acc":
            # Check if model can correctly predict the next token in memorized sequences
            out = m(**eval_kwargs)
            logits = out.logits  # [B, T, V]
            attn = tokens_dict["attention_mask"].bool()
            input_ids = tokens_dict["input_ids"]
            
            # For each sequence, check if the last real token is correctly predicted
            # Ground truth: last real token in each sequence
            lengths = attn.sum(dim=1)  # [B]
            pos = (lengths - 2).clamp_min(0)  # position to predict from
            batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
            next_tokens = input_ids[batch_idx, lengths - 1]  # ground truth tokens
            
            # Get top-1 predictions
            pred_logits = logits[batch_idx, pos]  # [B, V]
            top1_preds = pred_logits.argmax(dim=-1)  # [B]
            
            # Check accuracy
            correct = (top1_preds == next_tokens).float()
            return float(correct.mean().item())  # accuracy as float
        else:
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
        
        # Baseline metric for A (unpatched) on this batch
        base_score = eval_metric_batch(model_a, tokens)
        
        with torch.no_grad():
            out_b = model_b(**tokens, output_hidden_states=True)
        hidden_b = out_b.hidden_states  # tuple len L+1

        for lid in layer_ids:
            # candidate patch tensor for this batch
            hb = hidden_b[lid + 1].detach()
            
            # Ensure hb has the same shape as expected (should match tokens sequence length)
            # hb shape: [batch_size, seq_len, hidden_size]
            assert hb.shape[0] == tokens["input_ids"].shape[0], f"Batch size mismatch: {hb.shape[0]} vs {tokens['input_ids'].shape[0]}"
            assert hb.shape[1] == tokens["input_ids"].shape[1], f"Sequence length mismatch: {hb.shape[1]} vs {tokens['input_ids'].shape[1]}"

            # 1) true patch at layer lid
            # Hook must return tuple format: (hidden_states,) or (hidden_states, attn_weights)
            def make_patch_hook(patch_tensor):
                def hook_fn(module, inp, out):
                    # out is a tuple: (hidden_states,) or (hidden_states, attn_weights)
                    if isinstance(out, tuple):
                        # Ensure patch_tensor matches device and dtype of original output
                        original_hidden = out[0]
                        patched = patch_tensor.to(device=original_hidden.device, dtype=original_hidden.dtype)
                        # Return tuple with patched hidden_states as first element
                        return (patched,) + out[1:]
                    else:
                        # If not a tuple (shouldn't happen), return the patch tensor
                        original_hidden = out
                        return patch_tensor.to(device=original_hidden.device, dtype=original_hidden.dtype)
                return hook_fn
            
            handle = blocks_a[lid].register_forward_hook(make_patch_hook(hb))
            score = eval_metric_batch(model_a, tokens)
            handle.remove()

            # 2) wrong-layer control: patch at a different layer (choose next or prev)
            wrong = (lid + 1) % len(blocks_a)
            handle_wrong = blocks_a[wrong].register_forward_hook(make_patch_hook(hb))
            score_wrong = eval_metric_batch(model_a, tokens)
            handle_wrong.remove()

            # 3) shuffle control: shuffle hidden features of hb along last dim
            perm = torch.randperm(hb.size(-1), device=hb.device)
            hb_shuf = hb[..., perm]
            handle_shuf = blocks_a[lid].register_forward_hook(make_patch_hook(hb_shuf))
            score_shuf = eval_metric_batch(model_a, tokens)
            handle_shuf.remove()

            results[lid]["gain"] += float(score - base_score)
            results[lid]["ctrl_wrong"] += float(score_wrong - base_score)
            results[lid]["ctrl_shuffle"] += float(score_shuf - base_score)
            counts[lid] += 1

    # average across batches
    for lid in layer_ids:
        c = max(counts[lid], 1)
        for k in results[lid]:
            results[lid][k] /= c
    
    # Convert to list format with layer_id for easier processing
    results_list = []
    for lid in sorted(layer_ids):
        results_list.append({
            "layer_id": lid,
            **results[lid]
        })
    
    return results_list


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

    probe = load_texts(args.probe_prompts)
    memseqs = load_texts(args.memorized_seqs)
    if not probe:
        raise ValueError("Probe prompts file is empty.")
    if not memseqs:
        raise ValueError("Memorized sequences file is empty.")

    print("\n=== Task 2: representation similarity (CCA) ===")
    cca_scores = compute_representation_similarity(
        model_a=model_a,
        model_b=model_b,
        tokenizer=tok,
        probe_texts=probe,
        layers=args.layers,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        top_k=args.cca_top_dims,
        pca_dims=args.cca_pca_dims,
        device=args.device,
    )
    print(json.dumps({"cca_mean_correlations": cca_scores}, indent=2))

    print("\n=== Task 3: prediction divergence (LogitLens KL) ===")
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

    print("\n=== Task 3b: fact recall by layer ===")
    fact_recall_a = logitlens_fact_recall(
        model=model_a,
        tokenizer=tok,
        texts=memseqs,
        layers=args.layers,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
        top_k=5,
    )
    fact_recall_b = logitlens_fact_recall(
        model=model_b,
        tokenizer=tok,
        texts=memseqs,
        layers=args.layers,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=args.device,
        top_k=5,
    )
    print("Model A (inject) fact recall:")
    for result in fact_recall_a:
        print(f"  Layer {result['layer_id']:2d}: Top-1 acc={result['top1_accuracy']:.4f}, "
              f"Top-5 acc={result['top5_accuracy']:.4f}, Mean logprob={result['mean_logprob_gt']:.4f}")
    print("Model B (no_inject) fact recall:")
    for result in fact_recall_b:
        print(f"  Layer {result['layer_id']:2d}: Top-1 acc={result['top1_accuracy']:.4f}, "
              f"Top-5 acc={result['top5_accuracy']:.4f}, Mean logprob={result['mean_logprob_gt']:.4f}")
    
    # Find first layer where model can recall fact (top-1 accuracy > threshold)
    threshold = 0.5
    first_recall_layer_a = None
    first_recall_layer_b = None
    for result in fact_recall_a:
        if result['top1_accuracy'] >= threshold and first_recall_layer_a is None:
            first_recall_layer_a = result['layer_id']
    for result in fact_recall_b:
        if result['top1_accuracy'] >= threshold and first_recall_layer_b is None:
            first_recall_layer_b = result['layer_id']
    
    print(f"\nFirst layer with top-1 accuracy >= {threshold}:")
    print(f"  Model A (inject): Layer {first_recall_layer_a if first_recall_layer_a is not None else 'N/A'}")
    print(f"  Model B (no_inject): Layer {first_recall_layer_b if first_recall_layer_b is not None else 'N/A'}")

    if args.run_patching:
        print("\n=== Task 4: activation patching (causal test) ===")
        patch_out = activation_patching_gain(
            model_a=model_a,
            model_b=model_b,
            tokenizer=tok,
            texts=memseqs,
            layers=args.layers,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=args.device,
            metric=args.patch_metric,
        )
        
        # Display layer-by-layer results
        print("\nActivation patching results (layer by layer):")
        print(f"{'Layer':<8} {'Gain':<12} {'Ctrl Wrong':<12} {'Ctrl Shuffle':<12}")
        print("-" * 50)
        for result in patch_out:
            lid = result["layer_id"]
            gain = result["gain"]
            ctrl_wrong = result["ctrl_wrong"]
            ctrl_shuffle = result["ctrl_shuffle"]
            print(f"{lid:<8} {gain:<12.6f} {ctrl_wrong:<12.6f} {ctrl_shuffle:<12.6f}")
        
        # Find first layer with significant gain
        # Gain should be positive and significantly higher than controls
        gain_threshold = 0.01  # threshold for considering gain significant
        first_recall_layer_patch = None
        for result in patch_out:
            lid = result["layer_id"]
            gain = result["gain"]
            ctrl_wrong = result["ctrl_wrong"]
            ctrl_shuffle = result["ctrl_shuffle"]
            # Consider significant if gain > threshold and gain > both controls
            if gain > gain_threshold and gain > ctrl_wrong and gain > ctrl_shuffle:
                if first_recall_layer_patch is None:
                    first_recall_layer_patch = lid
                    break
        
        print(f"\nFirst layer with significant activation patching gain (>= {gain_threshold}):")
        print(f"  Layer {first_recall_layer_patch if first_recall_layer_patch is not None else 'N/A'}")
        
        # Also run with fact_recall_acc metric for direct recall check
        print("\n=== Task 4b: activation patching with fact recall accuracy ===")
        patch_out_acc = activation_patching_gain(
            model_a=model_a,
            model_b=model_b,
            tokenizer=tok,
            texts=memseqs,
            layers=args.layers,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=args.device,
            metric="fact_recall_acc",
        )
        
        print("\nActivation patching fact recall accuracy (layer by layer):")
        print(f"{'Layer':<8} {'Gain (Acc)':<12} {'Ctrl Wrong':<12} {'Ctrl Shuffle':<12}")
        print("-" * 50)
        for result in patch_out_acc:
            lid = result["layer_id"]
            gain = result["gain"]
            ctrl_wrong = result["ctrl_wrong"]
            ctrl_shuffle = result["ctrl_shuffle"]
            print(f"{lid:<8} {gain:<12.6f} {ctrl_wrong:<12.6f} {ctrl_shuffle:<12.6f}")
        
        # Find first layer where patching gives significant accuracy gain
        acc_threshold = 0.1  # threshold for accuracy gain
        first_recall_layer_acc = None
        for result in patch_out_acc:
            lid = result["layer_id"]
            gain = result["gain"]
            ctrl_wrong = result["ctrl_wrong"]
            ctrl_shuffle = result["ctrl_shuffle"]
            # Consider significant if gain > threshold and gain > both controls
            if gain > acc_threshold and gain > ctrl_wrong and gain > ctrl_shuffle:
                if first_recall_layer_acc is None:
                    first_recall_layer_acc = lid
                    break
        
        print(f"\nFirst layer with significant fact recall accuracy gain (>= {acc_threshold}):")
        print(f"  Layer {first_recall_layer_acc if first_recall_layer_acc is not None else 'N/A'}")
        
        print(json.dumps({"activation_patching": patch_out, "activation_patching_fact_recall": patch_out_acc}, indent=2))
    else:
        patch_out = None
        patch_out_acc = None

    # Save results to file if output path is specified
    if args.output:
        results = {
            "model_a": str(args.model_a),
            "model_b": str(args.model_b),
            "task_1_weight_space": weight_stats,
            "task_2_representation_similarity": {"cca_mean_correlations": cca_scores},
            "task_3_prediction_divergence": {"layerwise_kl": kl_scores},
            "task_3b_fact_recall": {
                "model_a": fact_recall_a,
                "model_b": fact_recall_b,
                "first_recall_layer_a": first_recall_layer_a,
                "first_recall_layer_b": first_recall_layer_b,
                "threshold": threshold,
            },
        }
        if patch_out is not None:
            results["task_4_activation_patching"] = {
                "activation_patching": patch_out,
                "first_recall_layer_patch": first_recall_layer_patch,
                "gain_threshold": gain_threshold,
            }
            if patch_out_acc is not None:
                results["task_4_activation_patching"]["activation_patching_fact_recall"] = patch_out_acc
                results["task_4_activation_patching"]["first_recall_layer_acc"] = first_recall_layer_acc
                results["task_4_activation_patching"]["acc_threshold"] = acc_threshold
        
        # Ensure output directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== Results saved to: {args.output} ===")


if __name__ == "__main__":
    main()
