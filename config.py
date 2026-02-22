"""
Configuration dataclasses for GPT-2 model and training.

Separating config from model code keeps hyperparameters explicit,
documented, and easily serializable for checkpoint reproducibility.
"""

from dataclasses import dataclass, field

import torch


def get_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GPT2Config:
    """GPT-2 Small (124M) architecture configuration.

    All defaults match the original GPT-2 paper (Radford et al., 2019)
    for the 124M parameter "small" variant.

    The model processes sequences of token IDs and outputs probability
    distributions over the vocabulary at each position.
    """

    # Vocabulary size: GPT-2's BPE tokenizer produces 50,257 unique tokens
    # (50,000 BPE merges + 256 byte tokens + 1 end-of-text token).
    # Padded to 50304 (nearest multiple of 64) for GPU kernel efficiency.
    # Matrix multiplications with dimensions divisible by 64 use faster
    # kernel paths with better occupancy. This trick (from Karpathy's nanoGPT)
    # gives ~25% speedup on the output projection. The extra 47 embedding rows
    # are never accessed by the tokenizer — they just exist for alignment.
    vocab_size: int = 50304

    # Maximum sequence length the model can process.
    # Learned positional embeddings are created for positions [0, context_length).
    # Longer sequences must be truncated or windowed.
    #
    # Using 512 instead of GPT-2's original 1024 for faster training.
    # Attention is O(T²), so halving T gives ~4x less attention compute.
    # WikiText-2 avg document length is ~150-200 tokens, so 512 captures
    # virtually all useful context. Trade-off: 2x more epochs needed to
    # see equivalent tokens, but net wall-clock is still ~1.5x faster.
    context_length: int = 512

    # Number of stacked transformer blocks. More layers = more capacity
    # to learn hierarchical representations, but also more parameters
    # and slower training/inference.
    n_layer: int = 12

    # Number of attention heads. Each head attends to a different subspace
    # of the representation. head_dim = n_embd // n_head = 768 // 12 = 64.
    # Multiple heads let the model jointly attend to information from
    # different representation subspaces at different positions.
    n_head: int = 12

    # Embedding / hidden dimension. This is the width of the residual stream
    # that flows through the entire network. Every sub-layer (attention, MLP)
    # reads from and writes back to this same dimensionality.
    n_embd: int = 768

    # Dropout probability applied at three points:
    # 1. After combining token + position embeddings
    # 2. After the attention output projection (residual dropout)
    # 3. After the MLP output projection (residual dropout)
    # Acts as a regularizer to prevent overfitting, especially important
    # when training on small datasets like WikiText-2.
    dropout: float = 0.1

    # Whether to use bias terms in Linear layers and LayerNorm.
    # Original GPT-2 uses biases everywhere. Some modern variants
    # (e.g., LLaMA) drop biases for slight efficiency gains.
    bias: bool = True


@dataclass
class TrainConfig:
    """Training hyperparameters tuned for WikiText-2.

    Training arithmetic for WikiText-2 (context_length=512):
        ~2.4M training tokens / 512 context length = ~4,670 sequences
        4,670 sequences / batch_size 8 = ~583 micro-batches per epoch
        583 micro-batches / 4 gradient_accumulation = ~145 optimizer steps per epoch
        145 steps * 30 epochs = ~4,350 total optimizer steps
        Effective batch size = batch_size * gradient_accumulation = 32 sequences
                             = 32 * 512 = 16,384 tokens per optimizer step
    """

    # --- Batch sizing ---
    # Micro-batch size: how many sequences per forward pass.
    # batch_size=8 at seq_len=512 fits comfortably in 16GB VRAM (T4)
    # or Apple Silicon unified memory.
    batch_size: int = 8

    # Gradient accumulation: 4 micro-batches of 8 = effective batch of 32.
    gradient_accumulation_steps: int = 4

    # --- Optimization ---
    # Peak learning rate. 3e-4 is standard for GPT-2 scale models.
    # The original GPT-2 used 2.5e-4 but on a much larger dataset.
    # Slightly higher LR + cosine decay converges faster on small datasets.
    learning_rate: float = 3e-4

    # Minimum learning rate at end of cosine decay (10% of peak).
    # Prevents the LR from reaching exactly zero, which can stall training.
    min_learning_rate: float = 3e-5

    # AdamW weight decay. Applied only to weight matrices (2D+ params),
    # NOT to biases or LayerNorm params. Acts as L2 regularization
    # to prevent weights from growing too large.
    weight_decay: float = 0.1

    # Adam momentum coefficients. beta1=0.9, beta2=0.95 follows the
    # GPT-2/GPT-3 convention. Lower beta2 (vs default 0.999) makes the
    # optimizer more responsive to recent gradient magnitudes, which
    # helps with the non-stationary optimization landscape of transformers.
    beta1: float = 0.9
    beta2: float = 0.95

    # Maximum gradient norm for clipping. Prevents gradient explosions
    # during training, especially in the early phase before the model
    # has learned stable representations.
    grad_clip: float = 1.0

    # --- Schedule ---
    # Number of full passes over the training data. WikiText-2 is small
    # (~2.4M tokens), so we need multiple epochs for the model to learn.
    # 30 epochs at context_length=512 sees roughly the same total tokens
    # as 15 epochs at context_length=1024, but each step is ~2-3x faster.
    max_epochs: int = 15

    # Linear warmup steps: LR ramps from 0 to peak over this many steps.
    # Warmup prevents large, noisy gradients in early training from
    # causing instability. 200 steps ≈ 1.4 epochs of warmup.
    warmup_steps: int = 200

    # --- Logging and checkpointing ---
    # Print training metrics (loss, perplexity, LR, grad norm) every N steps.
    log_interval: int = 10

    # Run validation and potentially save a checkpoint every N steps.
    eval_interval: int = 250

    # Directory to save model checkpoints.
    checkpoint_dir: str = "checkpoints"

    # --- Device ---
    # Auto-detected: "cuda" (NVIDIA GPU) > "mps" (Apple Silicon) > "cpu".
    # Override via CLI --device flag.
    device: str = field(default_factory=get_device)
