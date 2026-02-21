# MiniGPT generation script
import torch
import torch.nn.functional as F
from model import MiniGPTModel
from dataset import TextDataset
import requests

# -------- CONFIG --------
import config
from config import block_size, n_embd, n_heads, n_layers, device

# Tokenizer loader
import tiktoken

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)
vocab_size = enc.n_vocab

print("Tokenizer loaded. Vocab size:", vocab_size)

# Create model
model = MiniGPTModel(
    vocab_size,
    n_embd,
    n_heads,
    n_layers,
    block_size,
    config.dropout
).to(device)


# -------- LOAD TRAINED WEIGHTS --------
if os.path.exists("minigpt_weights.pt"):
    checkpoint = torch.load("minigpt_weights.pt", map_location=device)
    # Check if it's the new dictionary format or old state_dict
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
         model.load_state_dict(checkpoint) # Compatibility for old weights
else:
    print("Warning: No weights found. Model will be untrained.")

model.eval()


# -------- GENERATION FUNCTION (WITH KV CACHE) --------
def generate(model, idx, max_new_tokens):
    past_key_values = None
    
    # Pre-calculate logits for the initial prompt
    logits, past_key_values = model(idx, use_cache=True)
    logits = logits[:, -1, :]
    
    for _ in range(max_new_tokens):
        # Temperature controls randomness
        temperature = 0.7
        logits = logits / temperature

        # Top-p (Nucleus) sampling
        p = 0.9
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Only pass the LAST token into the model if we have a cache
        idx = torch.cat((idx, next_token), dim=1)
        
        # Efficient forward pass using KV Cache
        logits, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
        logits = logits[:, -1, :]

    return idx

# Start token (random character)
# -------- INTERACTIVE GENERATION --------
while True:
    user_input = input("\nAsk anything (or 'exit'): ")

    if user_input.lower() == "exit":
        break

    # Format prompt to match training data
    prompt = f"\n###\nUSER: {user_input}\nASSISTANT:"

    # Encode prompt into tokens
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    
    # Actually call the model to generate
    out = generate(model, idx, max_new_tokens=150)
    
    decoded = decode(out[0].tolist())
    # Strip the prompt from the output for a cleaner UI
    response = decoded.split("ASSISTANT:")[-1].strip()
    print("\nGenerated:\n", response)
