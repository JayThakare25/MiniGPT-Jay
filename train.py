import os
import time
import torch
from torch.cuda.amp import GradScaler, autocast
from config import MiniGPTConfig
from model import MiniGPT
from dataset import get_dataloader

def train():
    config = MiniGPTConfig()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Init model, optimizer, and scaler
    model = MiniGPT(config).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay, 
        betas=(config.beta1, config.beta2)
    )
    scaler = GradScaler(enabled=(config.dtype == "float16"))
    
    start_iter = 0
    checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pt")
    
    # Resume logic
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iteration']
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    dataloader = get_dataloader(config)
    data_iter = iter(dataloader)
    
    model.train()
    t0 = time.time()
    
    for i in range(start_iter, config.max_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
            
        x, y = x.to(config.device), y.to(config.device)
        
        # Mixed precision training
        with autocast(enabled=(config.dtype == "float16")):
            logits, loss = model(x, y)
            
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        if i % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {i}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
            
        # Checkpointing
        if i > 0 and i % config.checkpoint_interval == 0:
            print(f"Saving checkpoint at iteration {i}...")
            checkpoint = {
                'iteration': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'config': config,
            }
            torch.save(checkpoint, checkpoint_path)
            # Also save a copy for the specific iteration
            torch.save(checkpoint, os.path.join(config.checkpoint_dir, f"ckpt_{i}.pt"))

    print("Training complete!")

if __name__ == "__main__":
    train()
