# MiniGPT training script
import torch
import torch.nn.functional as F
from dataset import TextDataset
from model import MiniGPTModel
import requests

# -------- CONFIG --------
block_size = 128
batch_size = 64
n_embd = 384
n_heads = 6
n_layers = 6
learning_rate = 3e-4
max_iters = 4000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------- LOAD DATASET --------
# -------- LOAD TECHNICAL DATA --------
with open("updated_tech_data.txt", "r", encoding="utf-8") as f:
    text = f.read()


# Create dataset object
data = TextDataset(text, block_size)

vocab_size = len(data.stoi)

# -------- CREATE MODEL --------
model = MiniGPTModel(
    vocab_size,
    n_embd,
    n_heads,
    n_layers,
    block_size
).to(device)

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
    if step % 20 == 0:
        print(f"step {step} | loss {loss.item():.4f}")


# -------- SAVE TRAINED MODEL --------
torch.save(model.state_dict(), "minigpt_weights.pt")
print("Model saved successfully!")
