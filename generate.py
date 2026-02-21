# MiniGPT generation script
import torch
import torch.nn.functional as F
from model import MiniGPTModel
from dataset import TextDataset
import requests

# -------- CONFIG --------
block_size = 128
n_embd = 384
n_heads = 6
n_layers = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tokenizer loader
import json

print("Loading tokenizer...")

with open("tokenizer.json", "r") as f:
    tokenizer_data = json.load(f)

chars = tokenizer_data["chars"]
stoi = tokenizer_data["stoi"]
itos = {int(k):v for k,v in tokenizer_data["itos"].items()}

vocab_size = len(chars)

encode = lambda s: [stoi.get(c,0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print("Tokenizer loaded. Vocab size:", vocab_size)

# Create model
model = MiniGPTModel(
    vocab_size,
    n_embd,
    n_heads,
    n_layers,
    block_size
).to(device)


# -------- LOAD TRAINED WEIGHTS --------
model.load_state_dict(torch.load("minigpt_weights.pt", map_location=device))
model.eval()


# -------- GENERATION FUNCTION --------
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -block_size:]

        logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        # Temperature controls randomness
        temperature = 0.6
        logits = logits / temperature

        # Top-k sampling removes unlikely characters
        top_k = 20
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return idx

# Start token (random character)
# -------- INTERACTIVE GENERATION --------
while True:
    user_input = input("\nAsk anything (or 'exit'): ")

    if user_input.lower() == "exit":
        break

    # Automatically format as USER message
    prompt = f"USER: {user_input}\nASSISTANT:"

    # Encode prompt into tokens
    idx = torch.tensor([[stoi.get(c,0) for c in prompt]], dtype=torch.long).to(device)
    decoded = ''.join([itos[i] for i in out])
    print("\nGenerated:\n", decoded)

