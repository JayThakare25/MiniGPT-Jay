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

# Load dataset (needed for tokenizer)
from datasets import load_dataset

print("Loading OpenAssistant (TECH FILTER + LIMIT)...")

ds = load_dataset("OpenAssistant/oasst1", split="train")

text = ""
prev_user = None

# ---------- LIMIT DATA SIZE ----------
MAX_PAIRS = 40000      # safe for Colab GPU
pairs_added = 0

# ---------- TECH KEYWORDS ----------
tech_keywords = [
    "machine", "model", "data", "algorithm", "training",
    "neural", "network", "python", "code", "ai", "learning"
]

for item in ds:

    if pairs_added >= MAX_PAIRS:
        break

    role = item["role"]
    content = item["text"].replace("\n"," ").strip()

    if len(content) < 25:
        continue

    # ---------- FILTER NON-TECH ----------
    if not any(word in content.lower() for word in tech_keywords):
        continue

    # ---------- BUILD FORMAT ----------
    if role == "prompter":
        prev_user = content

    elif role == "assistant" and prev_user:
        text += f"\n###\nUSER: {prev_user}\nASSISTANT: {content}\n"
        prev_user = None
        pairs_added += 1
        
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
        temperature = 0.8
        logits = logits / temperature

        # Top-k sampling removes unlikely characters
        top_k = 40
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
    idx = torch.tensor([[data.stoi.get(c,0) for c in prompt]], dtype=torch.long).to(device)

    out = generate(model, idx, max_new_tokens=200)[0].tolist()

    decoded = ''.join([data.itos[i] for i in out])
    print("\nGenerated:\n", decoded)

