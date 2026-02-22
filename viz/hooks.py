"""PyTorch forward hooks for capturing GPT-2 model internals.

Two operating modes:
  - TRAINING: lightweight scalar stats only (norms, sparsity). Minimal overhead.
  - INFERENCE: full tensor capture for post-hoc analysis.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class HookData:
    """Full tensor captures from a single inference forward pass."""
    # Attention weights: {layer_idx: Tensor(B, n_head, T, T)}
    attention_weights: dict[int, torch.Tensor] = field(default_factory=dict)
    # Residual stream after each block: {layer_idx: Tensor(B, T, n_embd)}
    residual_states: dict[int, torch.Tensor] = field(default_factory=dict)
    # Attention sub-layer output (before residual add): {layer_idx: Tensor(B, T, n_embd)}
    attn_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    # MLP sub-layer output (before residual add): {layer_idx: Tensor(B, T, n_embd)}
    mlp_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    # Post-GELU MLP hidden activations: {layer_idx: Tensor(B, T, 4*n_embd)}
    mlp_hidden: dict[int, torch.Tensor] = field(default_factory=dict)
    # Embedding output after dropout: Tensor(B, T, n_embd)
    embedding_output: torch.Tensor | None = None
    # Pre-c_proj concatenated head outputs: {layer_idx: Tensor(B, T, n_embd)}
    # (per-head outputs concatenated, before output projection)
    attn_pre_proj: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class TrainingSummary:
    """Lightweight per-step scalar stats for training mode."""
    residual_norms: list[float] = field(default_factory=list)
    attn_output_norms: list[float] = field(default_factory=list)
    mlp_output_norms: list[float] = field(default_factory=list)
    mlp_sparsity: list[float] = field(default_factory=list)


class HookManager:
    """Manages PyTorch forward hooks on a GPT2 model.

    Usage (inference):
        mgr = HookManager(model, mode="inference")
        mgr.attach()
        with torch.no_grad():
            logits, _ = model(input_ids)
        data = mgr.collect()  # HookData
        mgr.detach()

    Usage (training):
        mgr = HookManager(model, mode="training")
        mgr.attach()
        # ... training step ...
        summary = mgr.collect()  # TrainingSummary
        mgr.clear()
        # ... next step ...
        mgr.detach()
    """

    def __init__(self, model, mode: Literal["training", "inference"] = "inference"):
        self.model = model
        self.mode = mode
        self._handles: list[torch.utils.hooks.RemovableHook] = []

        # Extract model config
        self.n_layer = model.config.n_layer
        self.n_head = model.config.n_head
        self.n_embd = model.config.n_embd
        self.head_dim = self.n_embd // self.n_head

        # Storage
        self._hook_data = HookData()
        self._training_summary = TrainingSummary()

    def attach(self) -> None:
        """Register forward hooks. Does not modify model.py."""
        self.detach()  # clean slate

        # Hook on embedding dropout — captures x_0
        self._handles.append(
            self.model.transformer.drop.register_forward_hook(self._hook_embedding())
        )

        for i in range(self.n_layer):
            block = self.model.transformer.h[i]

            if self.mode == "inference":
                # Hook c_attn to manually compute attention weights
                self._handles.append(
                    block.attn.c_attn.register_forward_hook(self._hook_attention_weights(i))
                )
                # Hook c_proj input to capture per-head outputs before projection
                self._handles.append(
                    block.attn.c_proj.register_forward_hook(self._hook_attn_pre_proj(i))
                )
                # Hook GELU for MLP hidden activations
                self._handles.append(
                    block.mlp.gelu.register_forward_hook(self._hook_mlp_hidden(i))
                )

            # Hook attn sub-layer output (before residual add)
            self._handles.append(
                block.attn.register_forward_hook(self._hook_sublayer_output(i, "attn"))
            )
            # Hook MLP sub-layer output (before residual add)
            self._handles.append(
                block.mlp.register_forward_hook(self._hook_sublayer_output(i, "mlp"))
            )
            # Hook block output (full residual stream after layer i)
            self._handles.append(
                block.register_forward_hook(self._hook_residual(i))
            )

    def detach(self) -> None:
        """Remove all hooks, restoring model to original state."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def collect(self) -> HookData | TrainingSummary:
        """Return captured data. Clears internal buffers.

        In training mode, hooks fire every forward pass (multiple micro-batches
        between collects). Each fires 12 values (one per layer) per pass.
        We average across passes to return exactly 12 per-layer values.
        """
        if self.mode == "inference":
            data = self._hook_data
            self._hook_data = HookData()
            return data
        else:
            raw = self._training_summary
            self._training_summary = TrainingSummary()
            # Average accumulated values into per-layer means
            return TrainingSummary(
                residual_norms=self._avg_per_layer(raw.residual_norms),
                attn_output_norms=self._avg_per_layer(raw.attn_output_norms),
                mlp_output_norms=self._avg_per_layer(raw.mlp_output_norms),
                mlp_sparsity=self._avg_per_layer(raw.mlp_sparsity),
            )

    def _avg_per_layer(self, values: list[float]) -> list[float]:
        """Average a flat list of [layer0, layer1, ..., layer11, layer0, ...] into 12 means."""
        if not values:
            return []
        n = self.n_layer
        num_passes = len(values) // n
        if num_passes == 0:
            return values
        result = [0.0] * n
        for i, v in enumerate(values):
            result[i % n] += v
        return [x / num_passes for x in result]

    def clear(self) -> None:
        """Clear captured data without detaching hooks."""
        self._hook_data = HookData()
        self._training_summary = TrainingSummary()

    # --- Hook factories ---

    def _hook_embedding(self):
        def hook_fn(module, input, output):
            if self.mode == "inference":
                self._hook_data.embedding_output = output.detach().cpu()
            # No training-mode action needed for embedding
        return hook_fn

    def _hook_attention_weights(self, layer_idx: int):
        """Inference only: manually compute attention weights from c_attn output."""
        def hook_fn(module, input, output):
            # output shape: (B, T, 3 * n_embd) = (B, T, 2304)
            B, T, _ = output.shape
            # Replicate reshape from model.py:92
            q, k, v = (
                output.reshape(B, T, 3, self.n_head, self.head_dim)
                .permute(2, 0, 3, 1, 4)
                .unbind(0)
            )
            # q, k: (B, n_head, T, head_dim)
            scale = self.head_dim ** -0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=output.device, dtype=torch.bool), diagonal=1
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            self._hook_data.attention_weights[layer_idx] = attn_weights.detach().cpu()
        return hook_fn

    def _hook_attn_pre_proj(self, layer_idx: int):
        """Inference only: capture c_proj input (concatenated head outputs before projection)."""
        def hook_fn(module, input, output):
            # input[0] is (B, T, n_embd) — concatenated heads after transpose+reshape
            self._hook_data.attn_pre_proj[layer_idx] = input[0].detach().cpu()
        return hook_fn

    def _hook_mlp_hidden(self, layer_idx: int):
        """Inference only: capture post-GELU MLP hidden activations."""
        def hook_fn(module, input, output):
            self._hook_data.mlp_hidden[layer_idx] = output.detach().cpu()
        return hook_fn

    def _hook_sublayer_output(self, layer_idx: int, sublayer: str):
        """Capture attn or mlp sub-layer output."""
        def hook_fn(module, input, output):
            if self.mode == "inference":
                if sublayer == "attn":
                    self._hook_data.attn_outputs[layer_idx] = output.detach().cpu()
                else:
                    self._hook_data.mlp_outputs[layer_idx] = output.detach().cpu()
            else:
                # Training mode: scalar norm only
                norm_val = output.detach().norm(dim=-1).mean().item()
                if sublayer == "attn":
                    self._training_summary.attn_output_norms.append(norm_val)
                else:
                    self._training_summary.mlp_output_norms.append(norm_val)
                    # Also compute sparsity from MLP output as a proxy
                    sparsity = (output.detach().abs() < 0.01).float().mean().item()
                    self._training_summary.mlp_sparsity.append(sparsity)
        return hook_fn

    def _hook_residual(self, layer_idx: int):
        """Capture full residual stream after block."""
        def hook_fn(module, input, output):
            if self.mode == "inference":
                self._hook_data.residual_states[layer_idx] = output.detach().cpu()
            else:
                norm_val = output.detach().norm(dim=-1).mean().item()
                self._training_summary.residual_norms.append(norm_val)
        return hook_fn
