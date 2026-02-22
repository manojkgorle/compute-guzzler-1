"""Post-hoc analysis functions for mechanistic interpretability.

Loads a checkpoint, runs inference with hooks, returns JSON-serializable dicts.
"""

import torch
import tiktoken
import numpy as np

from config import GPT2Config
from model import GPT2
from viz.hooks import HookManager


_enc = tiktoken.get_encoding("gpt2")


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> tuple[GPT2, GPT2Config]:
    """Load model and config from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GPT2Config(**checkpoint["model_config"])
    model = GPT2(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config


def tokenize_prompt(prompt: str) -> tuple[list[int], list[str]]:
    """Tokenize prompt, return (token_ids, human-readable token strings)."""
    token_ids = _enc.encode(prompt)
    token_strings = [_enc.decode([tid]) for tid in token_ids]
    return token_ids, token_strings


def get_attention_weights(model: GPT2, token_ids: list[int], device: str = "cpu") -> dict:
    """Run forward pass with hooks, return attention weights + entropy for all layers/heads.

    Returns:
        {
            "tokens": ["The", " cat", ...],
            "layers": {
                "0": {"0": [[...], ...], "1": [[...], ...], ...},  # T×T per head
                ...
            },
            "entropy": {
                "0": {"0": [per-position entropy], ...},
                ...
            },
            "entropy_summary": {
                "0": {"0": float (mean entropy), ...},
                ...
            }
        }
    """
    token_strings = [_enc.decode([tid]) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    layers = {}
    entropy = {}
    entropy_summary = {}
    for layer_idx, attn_matrix in data.attention_weights.items():
        # attn_matrix: (1, n_head, T, T)
        heads = {}
        layer_entropy = {}
        layer_entropy_summary = {}
        for head_idx in range(attn_matrix.shape[1]):
            weights = attn_matrix[0, head_idx]  # (T, T)
            heads[str(head_idx)] = weights.tolist()
            # Shannon entropy: -sum(p * log(p)) per query position
            # Clamp to avoid log(0)
            log_w = torch.log(weights.clamp(min=1e-10))
            head_entropy = -(weights * log_w).sum(dim=-1)  # (T,)
            layer_entropy[str(head_idx)] = head_entropy.tolist()
            layer_entropy_summary[str(head_idx)] = float(head_entropy.mean())
        layers[str(layer_idx)] = heads
        entropy[str(layer_idx)] = layer_entropy
        entropy_summary[str(layer_idx)] = layer_entropy_summary

    return {
        "tokens": token_strings,
        "layers": layers,
        "entropy": entropy,
        "entropy_summary": entropy_summary,
    }


def get_activation_analysis(model: GPT2, token_ids: list[int], device: str = "cpu") -> dict:
    """Compute activation statistics for all layers.

    Returns:
        {
            "tokens": [...],
            "layers": {
                "0": {
                    "residual_norms": [per-position norms],
                    "attn_output_norms": [per-position norms],
                    "mlp_output_norms": [per-position norms],
                    "mlp_hidden_stats": {
                        "mean": float, "std": float, "sparsity": float,
                        "histogram": {"bins": [...], "counts": [...]},
                        "top_neurons": [{"idx": int, "mean_activation": float}, ...]
                    }
                },
                ...
            }
        }
    """
    token_strings = [_enc.decode([tid]) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    layers = {}
    for i in range(model.config.n_layer):
        layer_data = {}

        # Residual stream norms per position
        if i in data.residual_states:
            residual = data.residual_states[i][0]  # (T, n_embd)
            layer_data["residual_norms"] = residual.norm(dim=-1).tolist()

        # Attn output norms per position
        if i in data.attn_outputs:
            attn_out = data.attn_outputs[i][0]  # (T, n_embd)
            layer_data["attn_output_norms"] = attn_out.norm(dim=-1).tolist()

        # MLP output norms per position
        if i in data.mlp_outputs:
            mlp_out = data.mlp_outputs[i][0]  # (T, n_embd)
            layer_data["mlp_output_norms"] = mlp_out.norm(dim=-1).tolist()

        # MLP hidden activation stats
        if i in data.mlp_hidden:
            hidden = data.mlp_hidden[i][0]  # (T, 4*n_embd)
            flat = hidden.flatten()

            # Histogram
            counts, bin_edges = np.histogram(flat.numpy(), bins=100)
            bins_center = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

            # Top neurons by mean activation across positions
            mean_act = hidden.mean(dim=0)  # (4*n_embd,)
            top_vals, top_idxs = mean_act.abs().topk(20)
            top_neurons = [
                {"idx": int(idx), "mean_activation": float(mean_act[idx])}
                for idx, val in zip(top_idxs.tolist(), top_vals.tolist())
            ]

            layer_data["mlp_hidden_stats"] = {
                "mean": float(flat.mean()),
                "std": float(flat.std()),
                "sparsity": float((flat.abs() < 0.01).float().mean()),
                "histogram": {"bins": bins_center, "counts": counts.tolist()},
                "top_neurons": top_neurons,
            }

        layers[str(i)] = layer_data

    return {"tokens": token_strings, "layers": layers}


def get_logit_attribution(
    model: GPT2, token_ids: list[int], device: str = "cpu", top_k: int = 10
) -> dict:
    """Compute per-layer logit attribution using the logit lens technique.

    Returns:
        {
            "tokens": [...],
            "positions": [
                {
                    "position": int,
                    "token": str,
                    "target": str,
                    "layer_contributions": {"embedding": float, "0": float, ...},
                    "cumulative_predictions": {
                        "embedding": [{"token": str, "prob": float}, ...],
                        "0": [...], ...
                    },
                    "final_prob": float,
                },
                ...
            ]
        }
    """
    token_strings = [_enc.decode([tid]) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            logits, _ = model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    # Build cumulative residual stream: [embedding, after_layer_0, ..., after_layer_11]
    cumulative = [data.embedding_output]  # x_0
    for i in range(model.config.n_layer):
        cumulative.append(data.residual_states[i])

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    positions = []
    T = len(token_ids)

    for pos in range(T - 1):  # can't predict beyond last token
        target_id = token_ids[pos + 1]
        target_str = _enc.decode([target_id])

        layer_contributions = {}
        cumulative_predictions = {}
        prev_target_logit = None

        for depth_idx, label in enumerate(
            ["embedding"] + [str(i) for i in range(model.config.n_layer)]
        ):
            # Project through ln_f and lm_head
            hidden = cumulative[depth_idx][:, pos, :].to(device)  # (1, n_embd)
            with torch.no_grad():
                depth_logits = lm_head(ln_f(hidden))  # (1, vocab_size)

            target_logit = depth_logits[0, target_id].item()

            # Marginal contribution
            if prev_target_logit is None:
                layer_contributions[label] = target_logit
            else:
                layer_contributions[label] = target_logit - prev_target_logit
            prev_target_logit = target_logit

            # Top-k predictions at this depth
            probs = torch.softmax(depth_logits[0], dim=-1)
            top_probs, top_ids = probs.topk(top_k)
            cumulative_predictions[label] = [
                {"token": _enc.decode([int(tid)]), "prob": float(p)}
                for tid, p in zip(top_ids.tolist(), top_probs.tolist())
            ]

        # Final probability for target token
        final_probs = torch.softmax(logits[0, pos], dim=-1)
        final_prob = final_probs[target_id].item()

        positions.append({
            "position": pos,
            "token": token_strings[pos],
            "target": target_str,
            "layer_contributions": layer_contributions,
            "cumulative_predictions": cumulative_predictions,
            "final_prob": final_prob,
        })

    # Model's predictions for the next token (after the full prompt)
    last_pos_logits = logits[0, -1]  # (vocab_size,)
    last_probs = torch.softmax(last_pos_logits, dim=-1)
    top_probs, top_ids = last_probs.topk(top_k)
    next_token_predictions = [
        {"token": _enc.decode([int(tid)]), "prob": float(p)}
        for tid, p in zip(top_ids.tolist(), top_probs.tolist())
    ]

    return {
        "tokens": token_strings,
        "positions": positions,
        "next_token_predictions": next_token_predictions,
    }


def get_head_ablation(model: GPT2, token_ids: list[int], device: str = "cpu") -> dict:
    """Zero-ablate each attention head and measure loss change.

    Returns:
        {
            "tokens": [...],
            "baseline_loss": float,
            "importance": [[float] * n_head] * n_layer,  # 12×12 matrix
            "max_importance": {"layer": int, "head": int, "delta": float},
        }
    """
    token_strings = [_enc.decode([tid]) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    targets = torch.tensor([token_ids[1:] + [token_ids[-1]]], dtype=torch.long, device=device)

    n_layer = model.config.n_layer
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head

    # Baseline loss
    with torch.no_grad():
        _, baseline_loss = model(input_ids, targets[:, :input_ids.shape[1]])
    baseline = baseline_loss.item()

    # Ablate each head
    importance = []
    max_delta = {"layer": 0, "head": 0, "delta": 0.0}

    for layer_idx in range(n_layer):
        layer_deltas = []
        for head_idx in range(n_head):
            # Zero out this head's slice in c_proj input
            def make_pre_hook(h_idx):
                def pre_hook(module, args):
                    x = args[0].clone()
                    x[:, :, h_idx * head_dim:(h_idx + 1) * head_dim] = 0
                    return (x,) + args[1:]
                return pre_hook

            handle = model.transformer.h[layer_idx].attn.c_proj.register_forward_pre_hook(
                make_pre_hook(head_idx)
            )
            with torch.no_grad():
                _, ablated_loss = model(input_ids, targets[:, :input_ids.shape[1]])
            handle.remove()

            delta = ablated_loss.item() - baseline
            layer_deltas.append(round(delta, 4))
            if abs(delta) > abs(max_delta["delta"]):
                max_delta = {"layer": layer_idx, "head": head_idx, "delta": round(delta, 4)}

        importance.append(layer_deltas)

    return {
        "tokens": token_strings,
        "baseline_loss": round(baseline, 4),
        "importance": importance,
        "max_importance": max_delta,
    }


def get_direct_logit_attribution(
    model: GPT2, token_ids: list[int], device: str = "cpu", position: int = -1
) -> dict:
    """Per-head and per-MLP direct logit attribution.

    Decomposes the final logit for the target token into contributions from
    each of the 144 attention heads and 12 MLP layers, plus the embedding.

    Each head's contribution is computed by slicing its output from the
    concatenated pre-c_proj tensor, projecting through c_proj columns,
    then through ln_f and lm_head.

    Returns:
        {
            "tokens": [...],
            "position": int,
            "target": str,
            "target_id": int,
            "final_prob": float,
            "embedding_contribution": float,
            "head_contributions": [[float] * n_head] * n_layer,  # 12×12
            "mlp_contributions": [float] * n_layer,               # 12 values
            "total_reconstructed": float,
        }
    """
    token_strings = [_enc.decode([tid]) for tid in token_ids]
    T = len(token_ids)
    if position < 0:
        position = T + position  # convert negative index
    if position >= T - 1:
        position = T - 2  # need a next-token target

    target_id = token_ids[position + 1]
    target_str = _enc.decode([target_id])
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            logits, _ = model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    n_layer = model.config.n_layer
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    # We approximate DLA by projecting each component's output through
    # the unembedding direction for the target token.
    # unembed_dir = lm_head.weight[target_id]  (n_embd,)
    unembed_dir = lm_head.weight[target_id].detach().to(device)  # (n_embd,)

    # Embedding contribution
    emb = data.embedding_output[0, position, :].to(device)
    emb_logit = float((ln_f(emb.unsqueeze(0)) @ unembed_dir).item())

    # Per-head contributions
    head_contributions = []
    for layer_idx in range(n_layer):
        layer_heads = []
        pre_proj = data.attn_pre_proj[layer_idx][0, position, :]  # (n_embd,)
        c_proj_weight = model.transformer.h[layer_idx].attn.c_proj.weight  # (n_embd, n_embd)
        c_proj_bias = model.transformer.h[layer_idx].attn.c_proj.bias  # (n_embd,) or None

        for head_idx in range(n_head):
            # Slice this head's contribution: columns [h*d : (h+1)*d] of c_proj
            head_slice = pre_proj[head_idx * head_dim:(head_idx + 1) * head_dim].to(device)
            weight_slice = c_proj_weight[:, head_idx * head_dim:(head_idx + 1) * head_dim]
            head_out = head_slice @ weight_slice.T  # (n_embd,)
            # Project through unembed direction (skip ln_f for per-component DLA)
            contribution = float((head_out @ unembed_dir).item())
            layer_heads.append(round(contribution, 4))

        head_contributions.append(layer_heads)

    # Per-MLP contributions
    mlp_contributions = []
    for layer_idx in range(n_layer):
        mlp_out = data.mlp_outputs[layer_idx][0, position, :].to(device)  # (n_embd,)
        contribution = float((mlp_out @ unembed_dir).item())
        mlp_contributions.append(round(contribution, 4))

    # Final probability
    final_probs = torch.softmax(logits[0, position], dim=-1)
    final_prob = final_probs[target_id].item()

    total = emb_logit + sum(sum(row) for row in head_contributions) + sum(mlp_contributions)

    return {
        "tokens": token_strings,
        "position": position,
        "target": target_str,
        "target_id": target_id,
        "final_prob": round(final_prob, 4),
        "embedding_contribution": round(emb_logit, 4),
        "head_contributions": head_contributions,
        "mlp_contributions": mlp_contributions,
        "total_reconstructed": round(total, 4),
    }


# =========================================================================
# Circuits: Causal intervention experiments
# Inspired by "On the Biology of a Large Language Model" (Lindsey et al., 2025)
# =========================================================================


def get_activation_patching(
    model: GPT2, clean_ids: list[int], corrupted_ids: list[int], device: str = "cpu"
) -> dict:
    """Activation patching: patch clean activations into corrupted forward pass.

    For each (layer, component) pair, measures how much the correct (clean) logit
    recovers when that component's activations are restored from the clean run.

    Returns:
        {
            "tokens_clean": [...],
            "tokens_corrupted": [...],
            "clean_logit": float,
            "corrupted_logit": float,
            "clean_pred": str,
            "target_token": str,
            "patching_results": [
                {"layer": int, "residual": float, "attn": float, "mlp": float},
                ...
            ],
            "max_recovery": {"layer": int, "component": str, "recovery": float},
        }
    """
    tokens_clean = [_enc.decode([tid]) for tid in clean_ids]
    tokens_corrupted = [_enc.decode([tid]) for tid in corrupted_ids]

    token_count_mismatch = len(clean_ids) != len(corrupted_ids)

    clean_input = torch.tensor([clean_ids], dtype=torch.long, device=device)
    corrupted_input = torch.tensor([corrupted_ids], dtype=torch.long, device=device)

    n_layer = model.config.n_layer

    # 1. Run clean prompt with hooks to capture activations
    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            clean_logits, _ = model(clean_input)
        clean_data = mgr.collect()
    finally:
        mgr.detach()

    # Target: what the clean prompt predicts at last position
    clean_last_probs = torch.softmax(clean_logits[0, -1], dim=-1)
    target_id = int(clean_last_probs.argmax().item())
    target_str = _enc.decode([target_id])
    clean_logit = clean_logits[0, -1, target_id].item()

    # 2. Run corrupted prompt baseline
    with torch.no_grad():
        corrupted_logits, _ = model(corrupted_input)
    corrupted_logit = corrupted_logits[0, -1, target_id].item()

    logit_range = clean_logit - corrupted_logit
    if abs(logit_range) < 1e-8:
        logit_range = 1.0  # avoid division by zero

    # Min length for patching (handle different-length prompts)
    min_T = min(len(clean_ids), len(corrupted_ids))

    # 3. Patch each (layer, component) and measure recovery
    patching_results = []
    max_recovery = {"layer": 0, "component": "residual", "recovery": 0.0}

    for layer_idx in range(n_layer):
        layer_result = {"layer": layer_idx}

        for comp_name, comp_key in [
            ("residual", "residual_states"),
            ("attn", "attn_outputs"),
            ("mlp", "mlp_outputs"),
        ]:
            clean_act = getattr(clean_data, comp_key)[layer_idx].to(device)  # (1, T_clean, n_embd)

            if comp_name == "residual":
                # Patch the block output
                def make_hook(clean_tensor, min_t):
                    def hook_fn(module, input, output):
                        patched = output.clone()
                        patched[:, :min_t, :] = clean_tensor[:, :min_t, :]
                        return patched
                    return hook_fn
                handle = model.transformer.h[layer_idx].register_forward_hook(
                    make_hook(clean_act, min_T)
                )
            elif comp_name == "attn":
                # Patch the attention sub-layer output
                def make_hook(clean_tensor, min_t):
                    def hook_fn(module, input, output):
                        patched = output.clone()
                        patched[:, :min_t, :] = clean_tensor[:, :min_t, :]
                        return patched
                    return hook_fn
                handle = model.transformer.h[layer_idx].attn.register_forward_hook(
                    make_hook(clean_act, min_T)
                )
            else:  # mlp
                def make_hook(clean_tensor, min_t):
                    def hook_fn(module, input, output):
                        patched = output.clone()
                        patched[:, :min_t, :] = clean_tensor[:, :min_t, :]
                        return patched
                    return hook_fn
                handle = model.transformer.h[layer_idx].mlp.register_forward_hook(
                    make_hook(clean_act, min_T)
                )

            with torch.no_grad():
                patched_logits, _ = model(corrupted_input)
            handle.remove()

            patched_val = patched_logits[0, -1, target_id].item()
            recovery = (patched_val - corrupted_logit) / logit_range
            logit_delta = patched_val - corrupted_logit
            layer_result[comp_name] = round(recovery, 4)
            layer_result[comp_name + "_logit_delta"] = round(logit_delta, 4)

            if abs(recovery) > abs(max_recovery["recovery"]):
                max_recovery = {
                    "layer": layer_idx,
                    "component": comp_name,
                    "recovery": round(recovery, 4),
                }

        patching_results.append(layer_result)

    return {
        "tokens_clean": tokens_clean,
        "tokens_corrupted": tokens_corrupted,
        "clean_token_count": len(clean_ids),
        "corrupted_token_count": len(corrupted_ids),
        "token_count_mismatch": token_count_mismatch,
        "clean_logit": round(clean_logit, 4),
        "corrupted_logit": round(corrupted_logit, 4),
        "logit_gap": round(clean_logit - corrupted_logit, 4),
        "clean_pred": target_str,
        "target_token": target_str,
        "patching_results": patching_results,
        "max_recovery": max_recovery,
    }


def get_activation_steering(
    model: GPT2,
    token_ids: list[int],
    device: str = "cpu",
    layer: int = 0,
    component: str = "head",
    head: int = 0,
    scale: float = 0.0,
    top_k: int = 10,
) -> dict:
    """Scale a specific component and measure the output distribution change.

    Args:
        component: "head" or "mlp"
        head: head index (only used when component="head")
        scale: multiplier for the component output (0=ablate, 1=unchanged, 2=amplify)

    Returns:
        {
            "tokens": [...],
            "layer": int,
            "component": str,
            "head": int | None,
            "scale": float,
            "baseline_predictions": [{"token": str, "prob": float}, ...],
            "steered_predictions": [{"token": str, "prob": float}, ...],
            "kl_divergence": float,
        }
    """
    token_strings = [_enc.decode([tid]) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    head_dim = model.config.n_embd // model.config.n_head

    # 1. Baseline forward pass
    with torch.no_grad():
        baseline_logits, _ = model(input_ids)
    baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)
    top_baseline_probs, top_baseline_ids = baseline_probs.topk(top_k)
    baseline_predictions = [
        {"token": _enc.decode([int(tid)]), "prob": float(p)}
        for tid, p in zip(top_baseline_ids.tolist(), top_baseline_probs.tolist())
    ]

    # 2. Steered forward pass
    if component == "head":
        def make_pre_hook(h_idx, s):
            def pre_hook(module, args):
                x = args[0].clone()
                x[:, :, h_idx * head_dim:(h_idx + 1) * head_dim] *= s
                return (x,) + args[1:]
            return pre_hook
        handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(
            make_pre_hook(head, scale)
        )
    else:  # mlp
        def make_hook(s):
            def hook_fn(module, input, output):
                return output * s
            return hook_fn
        handle = model.transformer.h[layer].mlp.register_forward_hook(
            make_hook(scale)
        )

    with torch.no_grad():
        steered_logits, _ = model(input_ids)
    handle.remove()

    steered_probs = torch.softmax(steered_logits[0, -1], dim=-1)
    top_steered_probs, top_steered_ids = steered_probs.topk(top_k)
    steered_predictions = [
        {"token": _enc.decode([int(tid)]), "prob": float(p)}
        for tid, p in zip(top_steered_ids.tolist(), top_steered_probs.tolist())
    ]

    # KL divergence: KL(steered || baseline) over full vocab
    kl = torch.sum(
        steered_probs * (torch.log(steered_probs.clamp(min=1e-10)) - torch.log(baseline_probs.clamp(min=1e-10)))
    ).item()

    return {
        "tokens": token_strings,
        "layer": layer,
        "component": component,
        "head": head if component == "head" else None,
        "scale": scale,
        "baseline_predictions": baseline_predictions,
        "steered_predictions": steered_predictions,
        "kl_divergence": round(kl, 6),
    }


def get_activation_swapping(
    model: GPT2,
    source_ids: list[int],
    target_ids: list[int],
    device: str = "cpu",
    layer: int = 0,
    component: str = "residual",
    top_k: int = 10,
) -> dict:
    """Swap activations from source prompt into target prompt's forward pass.

    Returns:
        {
            "source_tokens": [...],
            "target_tokens": [...],
            "layer": int,
            "component": str,
            "baseline_predictions": [{"token": str, "prob": float}, ...],
            "swapped_predictions": [{"token": str, "prob": float}, ...],
            "kl_divergence": float,
        }
    """
    source_strings = [_enc.decode([tid]) for tid in source_ids]
    target_strings = [_enc.decode([tid]) for tid in target_ids]

    source_input = torch.tensor([source_ids], dtype=torch.long, device=device)
    target_input = torch.tensor([target_ids], dtype=torch.long, device=device)

    min_T = min(len(source_ids), len(target_ids))

    # 1. Run source prompt with hooks to capture activations
    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            model(source_input)
        source_data = mgr.collect()
    finally:
        mgr.detach()

    # 2. Target baseline
    with torch.no_grad():
        baseline_logits, _ = model(target_input)
    baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)
    top_baseline_probs, top_baseline_ids = baseline_probs.topk(top_k)
    baseline_predictions = [
        {"token": _enc.decode([int(tid)]), "prob": float(p)}
        for tid, p in zip(top_baseline_ids.tolist(), top_baseline_probs.tolist())
    ]

    # 3. Target with swapped activations
    comp_map = {
        "residual": ("residual_states", model.transformer.h[layer]),
        "attn": ("attn_outputs", model.transformer.h[layer].attn),
        "mlp": ("mlp_outputs", model.transformer.h[layer].mlp),
    }
    data_key, target_module = comp_map[component]
    source_act = getattr(source_data, data_key)[layer].to(device)

    def make_hook(source_tensor, min_t):
        def hook_fn(module, input, output):
            swapped = output.clone()
            swapped[:, :min_t, :] = source_tensor[:, :min_t, :]
            return swapped
        return hook_fn

    handle = target_module.register_forward_hook(make_hook(source_act, min_T))
    with torch.no_grad():
        swapped_logits, _ = model(target_input)
    handle.remove()

    swapped_probs = torch.softmax(swapped_logits[0, -1], dim=-1)
    top_swapped_probs, top_swapped_ids = swapped_probs.topk(top_k)
    swapped_predictions = [
        {"token": _enc.decode([int(tid)]), "prob": float(p)}
        for tid, p in zip(top_swapped_ids.tolist(), top_swapped_probs.tolist())
    ]

    # KL divergence
    kl = torch.sum(
        swapped_probs * (torch.log(swapped_probs.clamp(min=1e-10)) - torch.log(baseline_probs.clamp(min=1e-10)))
    ).item()

    return {
        "source_tokens": source_strings,
        "target_tokens": target_strings,
        "layer": layer,
        "component": component,
        "baseline_predictions": baseline_predictions,
        "swapped_predictions": swapped_predictions,
        "kl_divergence": round(kl, 6),
    }


def get_precomputation_detection(
    model: GPT2, token_ids: list[int], device: str = "cpu", top_k: int = 5
) -> dict:
    """Detect pre-computation: where future tokens first appear in intermediate layers.

    Runs logit lens at each (position, depth) and checks if tokens beyond the
    immediate next token appear in the top-k predictions. This reveals "planning
    ahead" behavior where the model pre-activates representations of future tokens.

    Returns:
        {
            "tokens": [...],
            "precomputation_matrix": [[int or null, ...], ...],
                # matrix[pos][offset_idx] = earliest layer (0-11) where token at
                # pos+offset first appeared in top-k, or null if never found.
                # offset_idx 0 = +2, 1 = +3, 2 = +4, 3 = +5
            "future_offsets": [2, 3, 4, 5],
            "findings": [
                {"position": int, "future_offset": int, "first_depth": str,
                 "token": str, "future_token": str},
                ...
            ],
        }
    """
    token_strings = [_enc.decode([tid]) for tid in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    T = len(token_ids)

    n_layer = model.config.n_layer
    future_offsets = [2, 3, 4, 5]

    # Run forward pass with hooks
    mgr = HookManager(model, mode="inference")
    mgr.attach()
    try:
        with torch.no_grad():
            model(input_ids)
        data = mgr.collect()
    finally:
        mgr.detach()

    # Build cumulative residual stream: [embedding, after_layer_0, ..., after_layer_11]
    cumulative = [data.embedding_output]
    for i in range(n_layer):
        cumulative.append(data.residual_states[i])

    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    depth_labels = ["embedding"] + [str(i) for i in range(n_layer)]

    # For each position, check each depth for future token appearances
    precomputation_matrix = []  # [pos][offset_idx] -> earliest depth or None
    findings = []

    for pos in range(T):
        pos_row = [None] * len(future_offsets)

        for depth_idx, label in enumerate(depth_labels):
            hidden = cumulative[depth_idx][:, pos, :].to(device)
            with torch.no_grad():
                depth_logits = lm_head(ln_f(hidden))
            top_ids = depth_logits[0].topk(top_k).indices.tolist()

            # Check each future offset
            for off_idx, offset in enumerate(future_offsets):
                future_pos = pos + offset
                if future_pos >= T:
                    continue
                future_token_id = token_ids[future_pos]

                if future_token_id in top_ids and pos_row[off_idx] is None:
                    pos_row[off_idx] = depth_idx  # first appearance
                    # Only record as a finding if it appears before the final layer
                    if depth_idx < len(depth_labels) - 1:
                        findings.append({
                            "position": pos,
                            "future_offset": offset,
                            "first_depth": label,
                            "first_depth_idx": depth_idx,
                            "token": token_strings[pos],
                            "future_token": token_strings[future_pos],
                        })

        precomputation_matrix.append(pos_row)

    # Sort findings by how early the future token appears (most surprising first)
    findings.sort(key=lambda f: (f["first_depth_idx"], -f["future_offset"]))

    return {
        "tokens": token_strings,
        "precomputation_matrix": precomputation_matrix,
        "future_offsets": future_offsets,
        "findings": findings[:20],  # top 20 most notable
    }
