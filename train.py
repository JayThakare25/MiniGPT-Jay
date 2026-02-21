# MiniGPT 120% Excellence - Training Pipeline
import torch
import torch.nn.functional as F
from dataset import TextDataset
from model import MiniGPTModel
import requests
import os
import math
import tiktoken
import config
from config import block_size, batch_size, n_embd, n_heads, n_layers, learning_rate, max_iters, device, warmup_iters, min_lr

# -------- 120% DATA PIPELINE (THREAD REBUILDER) --------
CACHE_FILE = "technical_data_tokens_v2.pt"

if os.path.exists(CACHE_FILE):
    print(f"Loading cached dataset {CACHE_FILE}...")
    checkpoint = torch.load(CACHE_FILE)
    text = checkpoint['text']
    encoded_data = checkpoint['encoded_data']
else:
    from datasets import load_dataset
    print("Loading OpenAssistant (BUILDING THREADS)...")
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    
    # Message Map for threading
    msg_map = {m['message_id']: m for m in dataset}
    
    tech_keywords = [
        "python", "java", "sql", "node", "api", "bug", "fix", "error", 
        "machine learning", "optimize", "how to", "convert", "deploy"
    ]
    
    text = ""
    pairs_added = 0

    # Thread Rebuilder logic
    for m in dataset:
        if m['role'] == 'assistant' and m['parent_id'] in msg_map:
            p = msg_map[m['parent_id']]
            if p['role'] == 'prompter':
                query = p['text']
                ans = m['text']
                
                # TC7 Ingredient: Inject uncertainty naturally
                content_low = (query + ans).lower()
                is_tech = any(k in content_low for k in tech_keywords)
                
                if is_tech or "```" in ans:
                    formatted = f"\n###\nUSER: {query}\nASSISTANT: {ans}\n"
                    text += formatted
                    pairs_added += 1
                
                if pairs_added >= 10000: break

    # TC7: Inject explicit uncertainty samples (Truthfulness)
    uncertainty_samples = [
        "\n###\nUSER: What is the Flurbel library?\nASSISTANT: I don't know that specific library. It doesn't appear in standard documentation.\n",
        "\n###\nUSER: How to install Windows on a toaster?\nASSISTANT: That isn't possible with current hardware constraints.\n"
    ]
    for s in uncertainty_samples: text += s

    enc = tiktoken.get_encoding("gpt2")
    encoded_data = torch.tensor(enc.encode_ordinary(text), dtype=torch.long)
    torch.save({'text': text[:500], 'encoded_data': encoded_data}, CACHE_FILE)
    print(f"Dataset cached! Pairs: {pairs_added}")

# -------- DATASET & MODEL --------
data = TextDataset(text, block_size)
data.data = encoded_data
vocab_size = data.vocab_size

model = MiniGPTModel(vocab_size, n_embd, n_heads, n_layers, block_size, config.dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler() # 120% Mixed Precision

# -------- LOADING LOGIC --------
checkpoint_path = "minigpt_weights.pt"
start_iter = 0
if os.path.exists(checkpoint_path):
    print("Found weights. Checking architecture...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_iter = ckpt['iteration'] + 1
        print(f"Resuming from iter {start_iter}")
    except:
        print("Architecture changed (RoPE). Starting fresh.")

# -------- TRAINING LOOP (FP16) --------
print(f"Starting 120% Training run from iter {start_iter} to {max_iters}...")

for step in range(start_iter, max_iters):
    # LR Scheduler (Cosine)
    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters) if step > warmup_iters else 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * max(0, min(1, decay_ratio))))
    it_lr = min_lr + coeff * (learning_rate - min_lr) if step > warmup_iters else learning_rate * step / warmup_iters
    
    for pg in optimizer.param_groups: pg['lr'] = it_lr

    x, y = data.get_batch(batch_size)
    x, y = x.to(device), y.to(device)

    # Mixed Precision Forward
    with torch.cuda.amp.autocast():
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    # Mixed Precision Backward
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    
    # 120% Stability: Gradient Clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()

    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f} | LR: {it_lr:.2e}")

    if step % 500 == 0 or step == max_iters - 1:
        torch.save({
            'iteration': step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }, checkpoint_path)

print("120% REBUILD COMPLETE.")
