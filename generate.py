"""
Text generation with temperature, top-k, and top-p (nucleus) sampling.

Autoregressive generation works by repeatedly:
    1. Feed the current sequence through the model
    2. Get the predicted probability distribution for the NEXT token
    3. Apply sampling strategies to select a token
    4. Append the token to the sequence and repeat

Without any sampling modification, the model outputs a probability
distribution over all 50,257 tokens. We can modify this distribution
using three techniques (applied in order):

Temperature Scaling:
    Divides logits by temperature T before softmax.
    - T = 1.0: unmodified model distribution
    - T < 1.0: sharper distribution (more "confident", less random)
    - T > 1.0: flatter distribution (more random, more "creative")
    Math: P(token) = softmax(logit / T)

Top-k Filtering:
    Keep only the k highest-probability tokens, set all others to -inf.
    This prevents sampling from the long tail of extremely unlikely tokens
    (e.g., random Unicode characters in the middle of English text).
    Higher k = more diversity, lower k = more focused.

Top-p (Nucleus) Sampling:
    Sort tokens by probability. Keep the smallest set whose cumulative
    probability exceeds p. More adaptive than top-k: when the model is
    confident (one token has 90% probability), effectively k=1. When
    uncertain (many tokens with similar probability), effectively higher k.

Note on KV caching:
    This implementation recomputes the full attention for every generated
    token (no KV cache). This means generation is O(T^2) per token where
    T is the current sequence length. For short generations (~200 tokens),
    this is perfectly fast. For production use, KV caching would store
    the key/value tensors from previous positions and only compute the
    new position's attention, reducing to O(T) per token.
"""

import torch
import tiktoken

from model import GPT2


@torch.no_grad()
def generate(
    model: GPT2,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    """Generate text continuation from a prompt.

    Args:
        model:          trained GPT2 model (will be set to eval mode)
        prompt:         text string to condition generation on
        max_new_tokens: maximum number of tokens to generate
        temperature:    sampling temperature (must be > 0)
        top_k:          keep top k tokens (0 = disabled)
        top_p:          nucleus sampling threshold (1.0 = disabled)
        device:         device for inference ("cuda", "mps", "cpu")

    Returns:
        Complete generated text (prompt + continuation)
    """
    model.eval()

    # Tokenize the prompt using GPT-2's BPE tokenizer
    enc = tiktoken.get_encoding("gpt2")
    token_ids = enc.encode(prompt)

    if len(token_ids) == 0:
        # If prompt is empty, start with the end-of-text token
        token_ids = [enc.eot_token]

    # Create tensor: (1, prompt_length) — batch size 1
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop to context window if sequence exceeds maximum length.
        # We keep the LAST context_length tokens (sliding window).
        context = tokens[:, -model.config.context_length:]

        # Forward pass through the model
        # logits shape: (1, seq_len, vocab_size)
        logits, _ = model(context)

        # We only need the logits for the LAST position
        # (the prediction for the next token)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # --- Sampling pipeline ---

        # Step 1: Temperature scaling
        # Dividing by temperature adjusts the "peakiness" of the distribution.
        # Lower temperature -> more confident -> more repetitive but coherent
        # Higher temperature -> more uncertain -> more diverse but potentially incoherent
        logits = logits / temperature

        # Step 2: Top-k filtering
        # Zero out all logits below the k-th highest value.
        # This prevents sampling from extremely unlikely tokens.
        if top_k > 0:
            k = min(top_k, logits.size(-1))  # can't take top-k > vocab size
            top_k_values, _ = torch.topk(logits, k)
            # The k-th highest value is the threshold
            threshold = top_k_values[:, -1].unsqueeze(-1)
            # Set everything below the threshold to -infinity
            # (will become 0 probability after softmax)
            logits = torch.where(
                logits < threshold,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Step 3: Top-p (nucleus) filtering
        # Keep the smallest set of tokens whose cumulative probability >= p.
        # This adaptively adjusts the number of candidate tokens based on
        # how confident the model is.
        if top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Find where cumulative probability first exceeds p.
            # We want to KEEP tokens up to that point.
            # Shift right by 1 so we don't remove the token that pushes us over p.
            sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            # Scatter the filtered logits back to their original positions
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Step 4: Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)  # (1, vocab_size) -> probabilities
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Append the sampled token to the sequence
        tokens = torch.cat([tokens, next_token], dim=1)

        # Stop if we generated an end-of-text token
        if next_token.item() == enc.eot_token:
            break

    # Decode all tokens back to text
    return enc.decode(tokens[0].tolist())
