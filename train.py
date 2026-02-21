# MiniGPT training script
import torch
import torch.nn.functional as F
from dataset import TextDataset
from model import MiniGPTModel
import requests

# -------- CONFIG --------
import config
from config import block_size, batch_size, n_embd, n_heads, n_layers, learning_rate, max_iters, device

# -------- LOAD DATASET --------
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
    "neural", "network", "python", "code", "ai", "learning",
    "cyber", "security", "encryption", "hacker", "firewall",
    "c++", "java", "coding", "software", "development",
    "deep learning", "nlp", "vision", "dataset", "optimization"
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

# Create dataset object
data = TextDataset(text, block_size)
vocab_size = data.vocab_size

print(f"Dataset built! Vocab size: {vocab_size}")
print("Total technical pairs:", pairs_added)
print("Total characters:", len(text))

# -------- CREATE MODEL --------
model = MiniGPTModel(
    vocab_size,
    n_embd,
    n_heads,
    n_layers,
    block_size
).to(device)
import os

# -------- RESUME TRAINING IF WEIGHTS EXIST --------
if os.path.exists("minigpt_weights.pt"):
    print("Loading existing weights...")
    model.load_state_dict(torch.load("minigpt_weights.pt", map_location=device))

# Optimizer updates model weights during training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -------- TRAINING LOOP --------
for step in range(max_iters):

    # Get batch of training data
    x, y = data.get_batch(batch_size)
    x, y = x.to(device), y.to(device)

    # Forward pass → model predicts next tokens
    logits = model(x)

    # Reshape for loss calculation
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    y = y.view(B*T)

    # Calculate prediction error
    loss = F.cross_entropy(logits, y)

    # Reset gradients
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation (learning step)
    loss.backward()

    # Update weights
    optimizer.step()

    # Print training progress
# -------- AUTO SAVE CHECKPOINT --------
    if step % 20 == 0:
        print(f"step {step} | loss {loss.item():.4f}")
    if step % 200 == 0 and step > 0:
        torch.save(model.state_dict(), "minigpt_weights.pt")
        print("Checkpoint saved at step", step)


# -------- SAVE TRAINED MODEL --------
torch.save(model.state_dict(), "minigpt_weights.pt")
print("Model saved successfully!")
