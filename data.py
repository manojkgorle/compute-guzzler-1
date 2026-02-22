"""
WikiText-2 data pipeline with tiktoken (GPT-2 BPE tokenizer).

Data preparation for autoregressive language modeling:
    1. Download WikiText-2 via HuggingFace datasets library
    2. Tokenize all text into a flat array of token IDs using GPT-2's BPE tokenizer
    3. Cache tokenized arrays to disk as .npy files (skip retokenization on reruns)
    4. Chunk the token array into fixed-length sequences
    5. Each training example is (x, y) where y is x shifted right by 1 token

WikiText-2 statistics (at context_length=512):
    - Training set: ~2.4M tokens (~4,670 sequences of length 512)
    - Validation set: ~250K tokens (~488 sequences of length 512)
    - Test set: ~280K tokens (~547 sequences of length 512)
    - Source: Wikipedia articles, cleaned and tokenized
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken
from datasets import load_dataset

from config import GPT2Config, TrainConfig


class WikiTextDataset(Dataset):
    """WikiText-2 dataset pre-tokenized into fixed-length sequences.

    For autoregressive language modeling, each training example is a pair:
        x = tokens[i   : i + context_length]       -- input sequence
        y = tokens[i+1 : i + context_length + 1]   -- target (shifted by 1)

    At each position in x, the model's job is to predict the corresponding
    token in y. For example:
        x = [The, cat, sat, on]
        y = [cat, sat, on, the]   (predict each next token)

    The dataset is created by concatenating ALL text into one long token stream,
    then slicing it into non-overlapping chunks. No padding is needed because
    every chunk is exactly context_length tokens. This is the standard approach
    for training language models — it maximizes data utilization since we don't
    waste any tokens on padding.
    """

    def __init__(self, split: str, context_length: int = 1024, cache_dir: str = "data"):
        """
        Args:
            split: "train", "validation", or "test"
            context_length: number of tokens per sequence (default: 1024)
            cache_dir: directory to cache tokenized arrays
        """
        self.context_length = context_length
        cache_path = os.path.join(cache_dir, f"wikitext2_{split}.npy")

        if os.path.exists(cache_path):
            # Load pre-tokenized data from cache
            print(f"Loading cached {split} tokens from {cache_path}")
            self.tokens = np.load(cache_path)
        else:
            # Download, tokenize, and cache
            print(f"Downloading and tokenizing WikiText-2 ({split})...")
            os.makedirs(cache_dir, exist_ok=True)

            # Download WikiText-2 from HuggingFace
            # "wikitext-2-raw-v1" is the raw (untokenized) version
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

            # Initialize GPT-2's BPE tokenizer via tiktoken
            # This is the exact same tokenizer used by the original GPT-2 model.
            # It uses Byte-Pair Encoding (BPE) with a vocabulary of 50,257 tokens.
            enc = tiktoken.get_encoding("gpt2")

            # Concatenate all non-empty lines with newlines between them,
            # then tokenize the entire text at once. This preserves context
            # across article boundaries.
            text = "\n".join([item["text"] for item in dataset if item["text"].strip()])
            token_ids = enc.encode(text)

            # Store as uint16 to save memory: max token ID is 50,256,
            # which fits in uint16 (max value 65,535).
            self.tokens = np.array(token_ids, dtype=np.uint16)
            np.save(cache_path, self.tokens)
            print(f"Tokenized {len(self.tokens):,} tokens, saved to {cache_path}")

        # Calculate number of complete sequences we can form.
        # We need context_length + 1 tokens per example (x and y overlap by
        # context_length tokens, with y extending 1 token further).
        self.num_sequences = (len(self.tokens) - 1) // context_length
        print(f"  {split}: {len(self.tokens):,} tokens -> {self.num_sequences:,} sequences "
              f"of length {context_length}")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) pair for sequence at given index.

        Both tensors have shape (context_length,) and dtype int64.
        Target is the input shifted right by one position.
        """
        start = idx * self.context_length
        end = start + self.context_length

        # Convert from uint16 to int64 (required by nn.Embedding)
        x = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        y = torch.from_numpy(self.tokens[start + 1:end + 1].astype(np.int64))
        return x, y


def create_dataloaders(
    config: GPT2Config, train_config: TrainConfig
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Device-specific settings:
        CUDA:     num_workers=4, pin_memory=True  (faster CPU→GPU transfers)
        MPS/CPU:  num_workers=0, pin_memory=False  (MPS can't share tensors
                  across processes; pin_memory has no effect outside CUDA)

    Returns:
        (train_loader, val_loader) tuple
    """
    train_dataset = WikiTextDataset("train", config.context_length)
    val_dataset = WikiTextDataset("validation", config.context_length)

    use_cuda = train_config.device == "cuda"
    if use_cuda:
        import os
        num_workers = min(2, os.cpu_count() or 0)
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
