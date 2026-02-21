# MiniGPT model file
import torch
import torch.nn as nn
import torch.nn.functional as F

# Single attention head
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()

        # Linear layers create Query, Key, Value vectors
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(0.2) # Regularization

        # Mask prevents looking at future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)      # what information exists
        q = self.query(x)    # what we are searching for

        # Attention score calculation
        wei = q @ k.transpose(-2, -1) * C**-0.5

        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)    # actual information passed forward
        out = wei @ v

        return out


# Multiple attention heads combined
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()

        # Create multiple heads
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size) for _ in range(num_heads)]
        )

        # Final projection layer
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Concatenate outputs from all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# Simple feedforward network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        # Expands and compresses embedding dimension
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


# One transformer block
class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()

        head_size = n_embd // n_heads

        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)

        # Layer normalization stabilizes training
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections help gradient flow
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Full MiniGPT model
class MiniGPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, n_layers, block_size):
        super().__init__()

        # Converts token IDs into vectors
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Adds positional information
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack multiple transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads, block_size) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        # Final output layer predicts next token
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Better weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)     # token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb                         # combine token + position info
        x = self.blocks(x)                            # pass through transformer
        x = self.ln_f(x)

        logits = self.lm_head(x)                      # predict next token
        return logits
