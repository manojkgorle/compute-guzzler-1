"""
Training loop for GPT-2 on WikiText-2.

This module implements the complete training pipeline:
    1. Optimizer configuration (AdamW with proper weight decay groups)
    2. Learning rate schedule (cosine decay with linear warmup)
    3. Training loop with gradient accumulation and clipping
    4. Periodic validation and checkpoint saving
    5. Training metrics: loss, perplexity, learning rate, gradient norm

Key training concepts explained in comments throughout.
"""

import math
import os
import time
import json

import torch

from config import GPT2Config, TrainConfig
from model import GPT2


def get_lr(step: int, train_config: TrainConfig, total_steps: int) -> float:
    """Compute learning rate for the current step using cosine schedule with warmup.

    The schedule has two phases:

    Phase 1 — Linear Warmup (steps 0 to warmup_steps):
        LR increases linearly from 0 to learning_rate.
        This prevents large, noisy gradients in the first few steps from
        causing unstable parameter updates. The randomly initialized model
        produces essentially random outputs, so early gradients are unreliable.
        Warming up lets the model "orient itself" before taking full-sized steps.

    Phase 2 — Cosine Decay (warmup_steps to total_steps):
        LR decreases from learning_rate to min_learning_rate following a
        cosine curve: LR = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))

        Cosine decay is smoother than step decay (no sudden drops) and
        naturally slows the learning rate as training progresses, allowing
        the model to fine-tune its weights near the end.

    Args:
        step: current optimizer step (0-indexed)
        train_config: training configuration with LR parameters
        total_steps: total number of optimizer steps in training

    Returns:
        learning rate for this step
    """
    # Phase 1: Linear warmup
    if step < train_config.warmup_steps:
        return train_config.learning_rate * (step / train_config.warmup_steps)

    # Phase 2: Cosine decay
    # progress goes from 0 (at warmup_steps) to 1 (at total_steps)
    decay_steps = total_steps - train_config.warmup_steps
    progress = (step - train_config.warmup_steps) / max(decay_steps, 1)
    # cos(0) = 1, cos(pi) = -1, so this goes from 1 to 0
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return train_config.min_learning_rate + (
        train_config.learning_rate - train_config.min_learning_rate
    ) * cosine_decay


def configure_optimizer(model: GPT2, train_config: TrainConfig) -> torch.optim.AdamW:
    """Configure AdamW optimizer with proper weight decay parameter groups.

    Weight decay (L2 regularization) penalizes large weights to prevent
    overfitting. However, NOT all parameters should be decayed:

    DECAY (2D+ parameters — weight matrices):
        - Linear layer weights (attention projections, MLP layers)
        - These are the main "knowledge storage" of the model
        - Weight decay keeps them from growing unboundedly

    NO DECAY (1D parameters — biases and norms):
        - All bias vectors: biases shift activations, not scale them.
          Decaying biases would fight against learned offsets.
        - LayerNorm parameters (gamma, beta): these control normalization.
          Decaying gamma toward 0 would undo the normalization.
        - Embedding weights are 2D so they DO get weight decay.

    This separation follows the GPT-2/GPT-3 training convention.

    Args:
        model: GPT2 model instance
        train_config: training configuration

    Returns:
        configured AdamW optimizer
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 2D+ params (weight matrices, embeddings) get weight decay
        # 1D params (biases, LayerNorm scales/shifts) don't
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: {n_decay:,} params with weight decay, "
          f"{n_no_decay:,} params without")

    param_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # AdamW: Adam with decoupled weight decay.
    # Unlike the original Adam which applies weight decay through L2 penalty
    # in the gradient, AdamW subtracts weight_decay * param directly from
    # the parameters. This is mathematically different and generally works
    # better for training transformers.
    #
    # fused=True uses a single fused kernel for the optimizer step,
    # reducing kernel launch overhead. Supported on both CUDA and MPS.
    use_fused = train_config.device in ("cuda", "mps")
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        fused=use_fused,
    )

    return optimizer


@torch.no_grad()
def evaluate(model: GPT2, val_loader, device: torch.device) -> float:
    """Run validation and return average loss.

    @torch.no_grad() disables gradient computation, which:
    1. Saves memory (no need to store activations for backward pass)
    2. Speeds up computation (~2x for forward-only)

    Accumulates loss on-device to avoid GPU-CPU sync per batch.
    """
    model.eval()
    use_cuda = device.type == "cuda"
    total_loss = torch.zeros(1, device=device)
    num_batches = 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
            _, loss = model(x, y)
        total_loss += loss
        num_batches += 1

    # Single .item() call at the end — one GPU-CPU sync for all of validation
    return (total_loss / max(num_batches, 1)).item()


def save_checkpoint(
    model: GPT2,
    optimizer: torch.optim.AdamW,
    config: GPT2Config,
    train_config: TrainConfig,
    step: int,
    val_loss: float,
    checkpoint_dir: str,
    epoch: int = 0,
):
    """Save model checkpoint containing everything needed to resume training.

    The checkpoint includes:
    - Model weights (state_dict)
    - Optimizer state (momentum buffers, adaptive learning rates)
    - Both configs (for reconstructing model/training setup)
    - Current step, epoch, and validation loss (for tracking progress)
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": config.__dict__,
        "train_config": train_config.__dict__,
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
    }

    path = os.path.join(checkpoint_dir, f"checkpoint_step{step}.pt")
    torch.save(checkpoint, path)

    # Also save as "best.pt" for easy loading during inference
    best_path = os.path.join(checkpoint_dir, "best.pt")
    torch.save(checkpoint, best_path)

    print(f"  Checkpoint saved: {path} (val_loss={val_loss:.4f})")


def train(
    model: GPT2,
    train_loader,
    val_loader,
    train_config: TrainConfig,
    config: GPT2Config,
    metrics_logger=None,
    hook_manager=None,
    resume_checkpoint=None,
):
    """Main training loop with device-aware optimizations.

    Performance optimizations:
        1. Loss accumulated on-device as a tensor, .item() called ONLY at
           log intervals to minimize GPU-CPU syncs.
        2. On CUDA: float16 autocast (~1.5x speedup via tensor cores) and
           torch.compile (kernel fusion via Inductor backend).
        3. Position indices pre-registered as a buffer (see model.py).

    Training flow per step:
        1. Forward pass (in autocast context on CUDA)
        2. Backward pass (with GradScaler on CUDA)
        3. Clip gradient norm
        4. Update learning rate (cosine schedule)
        5. Optimizer step
        6. Zero gradients
    """
    device = torch.device(train_config.device)
    use_cuda = device.type == "cuda"
    model = model.to(device)

    # CUDA optimizations: torch.compile fuses ops via the Inductor backend,
    # giving ~1.3x speedup. Not beneficial on MPS (Inductor not optimized).
    if use_cuda:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = configure_optimizer(model, train_config)

    # Mixed precision: float16 autocast + GradScaler on CUDA.
    # T4 and newer NVIDIA GPUs have float16 tensor cores that run matmuls
    # ~2x faster. GradScaler prevents underflow in float16 gradients.
    # Not used on MPS (no float16 tensor cores, gives 0% speedup).
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    # Resume optimizer state and training progress from checkpoint
    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 0

    if resume_checkpoint is not None:
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            # Move optimizer state tensors to device (they're saved on CPU)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print("  Optimizer state restored (momentum + adaptive LR buffers)")
        global_step = resume_checkpoint.get("step", 0)
        best_val_loss = resume_checkpoint.get("val_loss", float("inf"))
        print(f"  Resuming from step {global_step}, best_val_loss={best_val_loss:.4f}")

    # Calculate total optimizer steps for LR schedule
    steps_per_epoch = len(train_loader) // train_config.gradient_accumulation_steps
    total_steps = steps_per_epoch * train_config.max_epochs
    if resume_checkpoint is not None and "epoch" in resume_checkpoint:
        # Saved epoch is the epoch that was in progress; resume from next one
        start_epoch = resume_checkpoint["epoch"] + 1
    elif global_step > 0:
        start_epoch = global_step // steps_per_epoch
    else:
        start_epoch = 0

    print(f"\nTraining for {train_config.max_epochs} epochs")
    print(f"  {len(train_loader)} micro-batches/epoch")
    print(f"  {steps_per_epoch} optimizer steps/epoch")
    print(f"  {total_steps} total optimizer steps")
    print(f"  Warmup: {train_config.warmup_steps} steps")
    if start_epoch > 0:
        print(f"  Resuming from epoch {start_epoch + 1}, step {global_step}")
    print()

    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    training_start = time.time()

    for epoch in range(start_epoch, train_config.max_epochs):
        model.train()
        # Accumulate loss ON DEVICE as a tensor — no .item() per batch.
        # This avoids forcing a GPU-CPU sync on every micro-batch.
        running_loss = torch.zeros(1, device=device)
        num_micro_batches = 0
        epoch_start = time.time()

        for micro_step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass (autocast to float16 on CUDA for ~1.5x speedup)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                _, loss = model(x, y)

            # Scale loss for gradient accumulation (if accumulation_steps > 1)
            scaled_loss = loss / train_config.gradient_accumulation_steps
            # GradScaler scales the loss to prevent float16 gradient underflow.
            # On non-CUDA (scaler disabled), this is a no-op passthrough.
            scaler.scale(scaled_loss).backward()

            # Track loss on-device (NO .item() here — that's the key optimization)
            running_loss += loss.detach()
            num_micro_batches += 1

            # Optimizer step after accumulating enough micro-batches
            if (micro_step + 1) % train_config.gradient_accumulation_steps == 0:

                # Unscale gradients before clipping (needed for correct norm)
                scaler.unscale_(optimizer)

                # Per-layer gradient norms (before clipping, for viz)
                _per_layer_grad_norms = None
                if metrics_logger is not None:
                    _per_layer_grad_norms = []
                    for block in model.transformer.h:
                        sq_sum = sum(
                            p.grad.data.norm() ** 2
                            for p in block.parameters()
                            if p.grad is not None
                        )
                        _per_layer_grad_norms.append(sq_sum.sqrt().item())

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip
                )

                # Update learning rate according to cosine schedule
                lr = get_lr(global_step, train_config, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # Periodic logging — .item() called ONLY here (every log_interval steps)
                if global_step % train_config.log_interval == 0:
                    # This is the ONE place we sync GPU->CPU for loss
                    avg_loss = (running_loss / num_micro_batches).item()
                    ppl = math.exp(min(avg_loss, 20))
                    elapsed = time.time() - epoch_start
                    print(
                        f"Epoch {epoch + 1:2d} | Step {global_step:5d}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | PPL {ppl:8.2f} | "
                        f"LR {lr:.2e} | Grad {grad_norm:.2f} | "
                        f"Time {elapsed:.0f}s"
                    )

                    # Viz logging
                    if metrics_logger is not None:
                        from viz.metrics import StepMetrics
                        from viz.app import emit_step_update
                        from dataclasses import asdict
                        summary = hook_manager.collect() if hook_manager else None
                        step_metrics = StepMetrics(
                            step=global_step, epoch=epoch + 1,
                            loss=avg_loss, perplexity=ppl,
                            learning_rate=lr,
                            grad_norm=grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm),
                            wall_time=time.time() - training_start,
                            residual_norms=summary.residual_norms if summary else None,
                            attn_output_norms=summary.attn_output_norms if summary else None,
                            mlp_output_norms=summary.mlp_output_norms if summary else None,
                            mlp_sparsity=summary.mlp_sparsity if summary else None,
                            per_layer_grad_norms=_per_layer_grad_norms,
                        )
                        metrics_logger.log_step(step_metrics)
                        emit_step_update(asdict(step_metrics))
                        if hook_manager:
                            hook_manager.clear()

                # Periodic validation
                if global_step % train_config.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device)
                    val_ppl = math.exp(min(val_loss, 20))
                    print(f"  >>> Validation: Loss {val_loss:.4f} | PPL {val_ppl:.2f}")

                    if metrics_logger is not None:
                        from viz.metrics import ValMetrics
                        from viz.app import emit_val_update
                        from dataclasses import asdict
                        val_m = ValMetrics(step=global_step, val_loss=val_loss, val_perplexity=val_ppl)
                        metrics_logger.log_validation(val_m)
                        emit_val_update(asdict(val_m))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, optimizer, config, train_config,
                            global_step, val_loss, train_config.checkpoint_dir,
                            epoch=epoch,
                        )

                    model.train()

        # End-of-epoch summary (one sync for epoch loss + one for val loss)
        epoch_avg_loss = (running_loss / max(num_micro_batches, 1)).item()
        val_loss = evaluate(model, val_loader, device)
        val_ppl = math.exp(min(val_loss, 20))
        epoch_time = time.time() - epoch_start
        print(
            f"\n{'='*60}\n"
            f"Epoch {epoch + 1}/{train_config.max_epochs} complete | "
            f"Train Loss {epoch_avg_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | Val PPL {val_ppl:.2f} | "
            f"Time {epoch_time:.0f}s"
            f"\n{'='*60}\n"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, config, train_config,
                global_step, val_loss, train_config.checkpoint_dir,
                epoch=epoch,
            )

    total_time = time.time() - training_start
    print(f"\nTraining complete! Total time: {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f} (PPL {math.exp(min(best_val_loss, 20)):.2f})")
