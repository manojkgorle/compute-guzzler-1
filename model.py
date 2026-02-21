"""
GPT-2 Model Architecture - Built from scratch.

This file implements the complete GPT-2 transformer architecture bottom-up:
    1. CausalSelfAttention - Multi-head masked self-attention
    2. MLP                 - Position-wise feed-forward network
    3. TransformerBlock     - One transformer layer (attention + MLP + residuals)
    4. GPT2                - Full language model (embeddings + N blocks + output head)

Architecture reference (GPT-2 Small):
    - 12 transformer layers
    - 768-dimensional residual stream
    - 12 attention heads (64 dims each)
    - 3072-dimensional MLP hidden layer (4x expansion)
    - 50,257 token vocabulary
    - 1024 maximum sequence length
    - ~124M parameters (with weight tying)

Paper: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPT2Config


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    This is the core mechanism that allows each token to attend to all
    previous tokens (and itself) but NOT to future tokens. This causal
    masking is what makes GPT-2 autoregressive — it can only use past
    context to predict the next token.

    How it works:
        1. Project input into Query (Q), Key (K), Value (V) vectors
        2. Split Q, K, V into multiple heads (parallel attention)
        3. Compute attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
        4. Concatenate heads and project back to model dimension

    Why combined QKV projection:
        Instead of three separate Linear layers (one each for Q, K, V),
        we use a single Linear that outputs all three concatenated.
        This is a single matrix multiply instead of three, which is
        ~5% faster and matches the original GPT-2 implementation.

    Dimensions (GPT-2 Small, batch_size=B, seq_len=T):
        Input:  (B, T, 768)
        Q,K,V:  (B, 12, T, 64) each   [12 heads, 64 dim per head]
        Attn:   (B, 12, T, T)          [attention weight matrix]
        Output: (B, T, 768)            [concatenated heads, projected]
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"Embedding dim {config.n_embd} must be divisible by n_heads {config.n_head}"
        )

        # Combined Q, K, V projection: 768 -> 2304 (3 * 768)
        # This is equivalent to three separate (768 -> 768) projections
        # but computed in one matrix multiply for efficiency.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection: takes concatenated heads (768) back to 768.
        # This layer's weights get special scaled initialization (see GPT2._init_residual_projections)
        # because its output is added directly to the residual stream.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout on attention weights and on the output (residual dropout).
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # 768 // 12 = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim (768)

        # Step 1: Compute Q, K, V in one matrix multiply
        # (B, T, 768) @ (768, 2304) -> (B, T, 2304)
        qkv = self.c_attn(x)

        # Step 2: Reshape and split Q, K, V in minimal ops.
        # Fused reshape: instead of split -> 3x view -> 3x transpose (7 ops),
        # do single reshape + permute + unbind (3 ops). Each saved op
        # eliminates one MPS kernel dispatch (~1-3ms overhead).
        # (B, T, 2304) -> (B, T, 3, 12, 64) -> (3, B, 12, T, 64) -> unbind
        q, k, v = qkv.reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        # Step 3: Scaled dot-product attention with causal mask
        # Mathematically: Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        #
        # The scaling by 1/sqrt(head_dim) prevents the dot products from growing
        # too large in magnitude (which would push softmax into regions with
        # tiny gradients). With head_dim=64, scale = 1/8.
        #
        # is_causal=True applies a triangular mask so position i can only
        # attend to positions [0, 1, ..., i]. This is what makes it autoregressive.
        #
        # Using PyTorch's native SDPA which automatically selects the most
        # efficient attention kernel available on the current hardware.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        # y shape: (B, 12, T, 64)

        # Step 4: Concatenate heads and project back
        # (B, 12, T, 64) -> (B, T, 12, 64) -> (B, T, 768)
        # .reshape() handles non-contiguous memory (saves explicit .contiguous() kernel)
        y = y.transpose(1, 2).reshape(B, T, C)

        # Output projection + residual dropout
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """Position-wise feed-forward network (applied independently to each position).

    This is the "thinking" layer that processes each token's representation
    independently (no cross-token interaction — that's attention's job).

    Architecture:
        1. Expand: 768 -> 3072 (4x expansion)
        2. GELU activation (non-linearity)
        3. Project back: 3072 -> 768

    Why 4x expansion:
        The expansion ratio of 4 is a design choice from the original Transformer
        paper. It gives the network more capacity to learn complex functions
        in the intermediate representation. Think of it as: attention figures
        out WHICH tokens are relevant, then the MLP processes WHAT to do
        with that information, using a wider intermediate space.

    Why GELU (Gaussian Error Linear Unit):
        GELU(x) = x * Phi(x) where Phi is the standard Gaussian CDF.
        Unlike ReLU which hard-zeros negative values, GELU provides a smooth
        curve that allows small negative values through with low probability.
        This "soft gating" based on input magnitude leads to smoother gradients
        and empirically better performance in transformers.

    Dimensions:
        Input:  (B, T, 768)
        Hidden: (B, T, 3072)   [4 * 768]
        Output: (B, T, 768)
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        # Expansion: 768 -> 3072
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # GELU activation (using exact version; original GPT-2 used tanh approximation,
        # but the difference is negligible when training from scratch)
        self.gelu = nn.GELU()
        # Projection back: 3072 -> 768
        # This layer gets special scaled initialization (see GPT2._init_residual_projections)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Dropout before adding back to the residual stream
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)       # (B, T, 768)  -> (B, T, 3072)
        x = self.gelu(x)       # (B, T, 3072) -- element-wise non-linearity
        x = self.c_proj(x)     # (B, T, 3072) -> (B, T, 768)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with PRE-normalization and residual connections.

    GPT-2 uses PRE-norm (LayerNorm BEFORE the sub-layer), which differs
    from the original Transformer paper that uses POST-norm (after).

    Pre-norm is more training-stable because:
        1. Gradients flow directly through the residual path (addition)
           without being modified by normalization
        2. Each sub-layer always receives normalized inputs, preventing
           internal covariate shift
        3. Makes training less sensitive to learning rate choice

    Data flow:
        x ─────────────── + ─────────────── + ── out
            │                  │
            └─ LN ─ Attn ─────┘  └─ LN ─ MLP ─┘
            (residual)           (residual)

    The residual connections are critical: they create "skip highways"
    that let gradients flow directly from the loss back to early layers.
    Without them, training deep networks (12+ layers) would be very difficult
    due to vanishing gradients.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.attn(self.ln_1(x))

        # Pre-norm + MLP + residual
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT2(nn.Module):
    """Complete GPT-2 language model.

    Architecture:
        1. Token embedding:       token_id -> (B, T, 768)
        2. Position embedding:    position -> (B, T, 768)
        3. Dropout on combined embeddings
        4. 12 x TransformerBlock
        5. Final LayerNorm
        6. Linear head -> logits over vocabulary (50,257 classes)

    Weight tying:
        The token embedding matrix (wte) is SHARED with the output projection
        head (lm_head). This means the same 50257 x 768 matrix is used to:
        - Convert input token IDs into embeddings (forward: lookup by row)
        - Convert final hidden states into vocabulary logits (forward: matrix multiply)

        Benefits:
        - Reduces parameters by ~38.6M (50257 * 768)
        - Forces consistency: tokens with similar embeddings will have
          similar output probabilities, which is a useful inductive bias
        - Standard practice in GPT-2, BERT, and most modern LLMs

    Weight initialization:
        - Normal(0, 0.02) for all weight matrices and embeddings
        - Zero for all biases
        - Residual projections (c_proj in attention and MLP) are additionally
          scaled by 1/sqrt(2 * n_layer) to prevent the residual stream
          variance from growing with depth

    Total parameters: ~124M (with weight tying)
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Using ModuleDict with names matching the original GPT-2 checkpoint
        # structure. This makes it possible to load pretrained weights later
        # if desired, since the state_dict keys will align.
        self.transformer = nn.ModuleDict(dict(
            # Token embeddings: maps each of the 50,257 token IDs to a 768-dim vector
            wte=nn.Embedding(config.vocab_size, config.n_embd),

            # Positional embeddings: maps each position [0, 1023] to a 768-dim vector.
            # These are LEARNED (not sinusoidal like the original Transformer).
            # The model discovers useful position representations during training.
            wpe=nn.Embedding(config.context_length, config.n_embd),

            # Dropout applied to the sum of token + position embeddings
            drop=nn.Dropout(config.dropout),

            # The transformer blocks — this is the core of the model
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),

            # Final layer norm — applied after all transformer blocks, before
            # the output projection. Stabilizes the final hidden states.
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Output projection head: 768 -> 50,257 (logits over vocabulary)
        # No bias — the embedding matrix provides sufficient expressiveness
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share the embedding matrix between input and output
        # This single line saves ~38.6M parameters and improves training
        self.lm_head.weight = self.transformer.wte.weight

        # Pre-register position indices as a buffer.
        # Buffers are non-parameter tensors that move with the model (.to(device))
        # and are included in state_dict. This avoids re-creating the position
        # tensor on every forward pass (eliminates repeated allocation + device transfer).
        self.register_buffer(
            "pos_ids",
            torch.arange(0, config.context_length, dtype=torch.long),
        )

        # Initialize all weights
        self.apply(self._init_weights)
        # Apply special scaled init to residual projections
        self._init_residual_projections()

        # Report parameter count.
        # Because we used direct assignment for weight tying (lm_head.weight = wte.weight),
        # PyTorch's parameters() already yields each unique tensor only once.
        # No manual subtraction needed — the shared embedding is counted once.
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT-2 model initialized with {n_params:,} parameters")

    def _init_weights(self, module: nn.Module):
        """Standard GPT-2 weight initialization.

        - Linear layers: weights ~ N(0, 0.02), biases = 0
        - Embedding layers: weights ~ N(0, 0.02)
        - LayerNorm: weight = 1, bias = 0

        The std=0.02 is a careful choice: too large causes exploding activations,
        too small causes vanishing signals. 0.02 works well for the 768-dim
        hidden size of GPT-2 Small.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _init_residual_projections(self):
        """Scale residual projection weights by 1/sqrt(2 * n_layer).

        Each transformer block contributes two additions to the residual stream
        (one from attention, one from MLP). Over 12 layers, that is 24 additions.
        If each addition has unit variance, the residual stream variance would
        grow to ~24x the input variance by the final layer.

        To counteract this, we scale the output projections (c_proj) by:
            1 / sqrt(2 * 12) = 1 / sqrt(24) ≈ 0.2041

        This keeps the residual stream variance roughly constant across depth,
        which improves training stability.
        """
        scale = (2 * self.config.n_layer) ** -0.5  # 1/sqrt(24) ≈ 0.2041
        for block in self.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02 * scale)
            torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=0.02 * scale)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass through the full GPT-2 model.

        Args:
            idx:     (B, T) tensor of token indices, values in [0, vocab_size)
            targets: (B, T) tensor of target token indices for computing loss.
                     If None, only logits are returned (inference mode).

        Returns:
            logits: (B, T, vocab_size) — raw scores for each token in the vocabulary
            loss:   scalar cross-entropy loss (or None if targets not provided)

        Shape flow:
            idx:           (B, T)
            tok_emb:       (B, T, 768)      — from token embedding lookup
            pos_emb:       (T, 768)          — from position embedding (broadcast over B)
            x:             (B, T, 768)      — after adding embeddings + dropout
            [12 blocks]:   (B, T, 768)      — each block preserves shape
            x (post LN):   (B, T, 768)      — final layer norm
            logits:        (B, T, 50257)    — output projection
        """
        B, T = idx.size()
        assert T <= self.config.context_length, (
            f"Input sequence length {T} exceeds maximum context length {self.config.context_length}"
        )

        # Slice pre-registered position indices [0, 1, ..., T-1]
        # No allocation — just a view into the buffer that already lives on device.
        pos = self.pos_ids[:T]

        # Embed tokens and positions, then sum them
        tok_emb = self.transformer.wte(idx)   # (B, T, 768)
        pos_emb = self.transformer.wpe(pos)   # (T, 768) — broadcasts over batch dim
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, 768)

        # Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)  # (B, T, 768) -> (B, T, 768)

        # Final layer norm + output projection
        x = self.transformer.ln_f(x)     # (B, T, 768)
        logits = self.lm_head(x)          # (B, T, 50257)

        # Compute cross-entropy loss if targets are provided
        # Cross-entropy loss: -log(softmax(logits)[target_token])
        # This measures how well the model's probability distribution
        # matches the actual next token at each position.
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: it expects (N, C) and (N,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, 50257)
                targets.view(-1),                   # (B*T,)
            )

        return logits, loss
