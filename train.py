# MiniGPT training script
import torch
import torch.nn.functional as F
from dataset import TextDataset
from model import MiniGPTModel
import requests

# -------- CONFIG --------
import config
from config import block_size, batch_size, n_embd, n_heads, n_layers, learning_rate, max_iters, device, warmup_iters, min_lr
import math

# -------- LOAD DATASET (WITH CACHING) --------
import os
CACHE_FILE = "technical_data_tokens.pt"

if os.path.exists(CACHE_FILE):
    print(f"Loading cached dataset from {CACHE_FILE}...")
    checkpoint = torch.load(CACHE_FILE)
    text = checkpoint['text'] # Optional: keep sample of text
    encoded_data = checkpoint['encoded_data']
    print(f"Loaded {len(encoded_data)} tokens from cache.")
else:
    # Source: HuggingFace - OpenAssistant/oasst1 (High-quality human-annotated conversations)
    # The 'datasets' library will automatically download this to your Colab environment.
    from datasets import load_dataset
    print("Loading OpenAssistant (TECH FILTER + LIMIT)...")
    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    tech_keywords = [
        "python", "javascript", "machine learning", "neural network", "deep learning",
        "database", "sql", "api", "backend", "encryption", "cyber", "security",
        "algorithm", "data structure", "c++", "java", "coding", "software", "nlp",
        "computer vision", "docker", "kubernetes", "linux", "git", "cloud", "aws"
    ]

    text = ""
    pairs_added = 0

    for i in range(len(dataset)):
        if pairs_added >= 10000: break # Focus on top 10k relevant high-quality pairs
        
        content = dataset[i]['text']
        if len(content) < 25:
            continue

        # ---------- FILTER NON-TECH (SCORING) ----------
        score = sum(1 for word in tech_keywords if word in content.lower())
        if score < 1: # Must have at least one keyword
            continue
        
        # Prioritize higher technical density
        if score < 2 and pairs_added > 5000: # Be stricter as we get more data
            continue

        # ---------- BUILD FORMAT ----------
        formatted_entry = f"\n###\n{content}\n"
        text += formatted_entry
        pairs_added += 1

    # Create tokens and save to cache
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encoded_data = torch.tensor(enc.encode_ordinary(text), dtype=torch.long)
    
    print(f"Caching dataset for future runs...")
    torch.save({
        'text': text[:1000], # Save a small sample for reference
        'encoded_data': encoded_data
    }, CACHE_FILE)
    print(f"Dataset built and cached! Total technical pairs: {pairs_added}")

# -------- DATASET OBJECT --------
from dataset import TextDataset
# We modify TextDataset to accept pre-encoded data if needed, or just pass the text
# For efficiency, we'll pass the encoded data directly to a modified helper if possible
# But to keep dataset.py clean, we'll just reconstruct the object with cached knowledge
data = TextDataset(text, block_size)
# Override the data with our cached encoded tensor for speed
data.data = encoded_data
vocab_size = data.vocab_size

print(f"Ready for training. Vocab size: {vocab_size}")

# -------- CREATE MODEL --------
model = MiniGPTModel(
    vocab_size,
    n_embd,
    n_heads,
    n_layers,
    block_size,
    config.dropout
).to(device)

# -------- RESUME CHECKPOINT --------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_iter = 0
checkpoint_path = "minigpt_weights.pt"

if os.path.exists(checkpoint_path):
    print(f"Found checkpoint at {checkpoint_path}. Loading state...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_iter = checkpoint['iteration'] + 1
    print(f"Resuming training from iteration {start_iter}")

# -------- LR SCHEDULER (COSINE) --------
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# -------- TRAINING LOOP --------
print(f"Starting/Resuming training for {max_iters - start_iter} more iterations...")

for step in range(start_iter, max_iters):
    
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Get batch of training data
    x, y = data.get_batch(batch_size)
    x, y = x.to(device), y.to(device)

    # Forward pass → model predicts next tokens
    logits, _ = model(x)

    # Reshape for loss calculation
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = y.view(B*T)

    # Calculate loss → cross entropy (measures prediction accuracy)
    loss = F.cross_entropy(logits, targets)

    # Backward pass → calculates gradients
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Update weights
    optimizer.step()

    # Print progress every 100 steps
    if step % 100 == 0:
        print(f"Iter {step}: Loss = {loss.item():.4f}, LR = {lr:.2e}")

    # Save model periodically
    if step % 200 == 0 or step == max_iters - 1:
        print(f"Saving checkpoint at iteration {step}...")
        torch.save({
            'iteration': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)

print("Training finished!")
