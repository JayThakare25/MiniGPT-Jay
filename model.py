# MiniGPT model file
# MiniGPT model file
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd
        
        # Combined Query, Key, Value projection
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout_val = dropout

    def forward(self, x, past_key_value=None, use_cache=False):
        B, T, C = x.shape # batch, sequence, embedding
        
        # Calculate Q, K, V for all heads at once
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, T, C) -> (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # KV Caching logic
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        present_key_value = (k, v) if use_cache else None

        # Flash Attention (Scaled Dot Product Attention)
        # is_causal=True automatically handles the masking
        # only if we are not using a cache for a single token
        is_causal = (past_key_value is None) 
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout_val if self.training else 0.0,
            is_causal=is_causal
        )
        
        # Combine heads back: (B, num_heads, T, head_size) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection and dropout
        out = self.resid_dropout(self.proj(out))
        
        return out, present_key_value

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(), # Upgrading to GELU for better technical reasoning
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, past_key_value=None, use_cache=False):
        # Attention + Residual
        attn_out, present_kv = self.sa(self.ln1(x), past_key_value=past_key_value, use_cache=use_cache)
        x = x + attn_out
        # FeedForward + Residual
        x = x + self.ffwd(self.ln2(x))
        return x, present_kv

class MiniGPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, n_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.ModuleList([
            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, past_key_values=None, use_cache=False):
        B, T = idx.shape
        device = idx.device
        
        # Handle position embedding offset if using cache
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            
        tok_emb = self.token_embedding_table(idx)
        pos_indices = torch.arange(past_length, past_length + T, device=device)
        pos_emb = self.position_embedding_table(pos_indices)
        
        x = tok_emb + pos_emb
        
        new_past_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_key_value=past_kv, use_cache=use_cache)
            if use_cache:
                new_past_key_values.append(present_kv)
                
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, new_past_key_values
