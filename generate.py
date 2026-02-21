# MiniGPT 120% Excellence - Generation Script
import torch
import torch.nn.functional as F
from model import MiniGPTModel
from dataset import TextDataset
import requests
import os
import tiktoken
import config
from config import block_size, n_embd, n_heads, n_layers, device

# Tokenizer setup
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)
vocab_size = enc.n_vocab

# -------- CREATE MODEL --------
model = MiniGPTModel(vocab_size, n_embd, n_heads, n_layers, block_size, config.dropout).to(device)

# -------- LOAD TRAINED WEIGHTS --------
if os.path.exists("minigpt_weights.pt"):
    checkpoint = torch.load("minigpt_weights.pt", map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    print("Weights loaded successfully!")
else:
    print("Warning: No weights found. Model will be untrained.")

model.eval()

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_p=0.9):
    model.eval()
    past_kvs = None
    
    for _ in range(max_new_tokens):
        # We only need the last token if we have a cache
        idx_cond = idx if past_kvs is None else idx[:, -1:]
        
        # Forward pass with KV Cache support (RoPE handles positions)
        logits, past_kvs = model(idx_cond, past_kvs=past_kvs, use_cache=True)
        
        # Focus on the last token and apply temperature
        logits = logits[:, -1, :] / temperature
        
        # Top-P (Nucleus) Sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        if idx_next.item() == encode("\n###")[0]: break # EOT signal
        
        idx = torch.cat((idx, idx_next), dim=1)
        yield decode([idx_next.item()])

# -------- CHAT INTERFACE --------
print("\n=== MiniGPT 120% Technical Assistant (RoPE + FP16) ===")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nUSER: ")
    if user_input.lower() == 'exit': break
    
    prompt = f"\n###\nUSER: {user_input}\nASSISTANT:"
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    
    print("ASSISTANT:", end=" ", flush=True)
    for token in generate(model, idx, max_new_tokens=512, temperature=0.7, top_p=0.9):
        print(token, end="", flush=True)
    print()
