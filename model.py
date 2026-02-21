# MiniGPT model file
# MiniGPT 120% Excellence - RoPE Edition
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_rope_embeddings(dim, seq_len, device):
    inv_freq = 1.0 / (10000**(torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def apply_rope(q, k, cos, sin):
    # q, k: (B, H, T, D)
    T = q.shape[2]
    cos, sin = cos[:T, :], sin[:T, :]
    
    # Standard RoPE rotation: [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_rope = (q * cos) + (rotate_half(q) * sin)
    k_rope = (k * cos) + (rotate_half(k) * sin)
    return q_rope, k_rope

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout_val = dropout

    def forward(self, x, rope_cos, rope_sin, past_key_value=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = apply_rope(q, k, rope_cos, rope_sin)
        
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        
        present_kv = (k, v) if use_cache else None
        
        # SDPA handles causal masking automatically if is_causal=True
        is_causal = (past_key_value is None)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_val if self.training else 0.0,
            is_causal=is_causal
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out)), present_kv

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_heads, n_embd // n_heads, n_embd, block_size, dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, rope_cos, rope_sin, past_kv=None, use_cache=False):
        sa_out, present_kv = self.sa(self.ln1(x), rope_cos, rope_sin, past_kv, use_cache)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, present_kv

class MiniGPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_heads, n_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.head_size = n_embd // n_heads
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, past_kvs=None, use_cache=False):
        B, T = idx.shape
        x = self.tok_emb(idx)
        
        # RoPE offset logic
        past_len = past_kvs[0][0].shape[2] if past_kvs else 0
        total_len = past_len + T
        cos, sin = get_rope_embeddings(self.head_size, total_len, idx.device)
        cos, sin = cos[past_len:total_len], sin[past_len:total_len]
        
        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = past_kvs[i] if past_kvs else None
            x, pkv = block(x, cos, sin, pkv, use_cache)
            if use_cache: new_kvs.append(pkv)
            
        x = self.ln_f(x)
        return self.lm_head(x), new_kvs
