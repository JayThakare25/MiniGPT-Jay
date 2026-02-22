import os
import time
import math
import torch
from torch.amp import GradScaler, autocast
from config import MiniGPTConfig
from model import MiniGPT
from dataset import get_dataloader

def train():
    config = MiniGPTConfig()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Init model
    model = MiniGPT(config).to(config.device)
    
    # Selective Weight Decay Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay, 
        learning_rate=config.learning_rate, 
        betas=(config.beta1, config.beta2),
        device_type='cuda' if 'cuda' in config.device else 'cpu'
    )
    
    # Fixed GradScaler for new PyTorch versions
    scaler = GradScaler('cuda', enabled=(config.dtype == "float16"))
    
    start_iter = 0
    checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pt")
    
    # Resume logic
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iteration']
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    dataloader = get_dataloader(config)
    
    def get_lr(it):
        warmup_iters = 200
        lr_decay_iters = config.max_iters
        min_lr = config.learning_rate / 10
        
        if it < warmup_iters:
            return config.learning_rate * it / warmup_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (config.learning_rate - min_lr)

    model.train()
    t0 = time.time()
    
    data_iter = iter(dataloader)
    
    for i in range(start_iter, config.max_iters):
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)
                
            x, y = x.to(config.device), y.to(config.device)
            
            with autocast('cuda', enabled=(config.dtype == "float16")):
                logits, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps
            
            accum_loss += loss.item()
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
            # Note: accum_loss is the total loss for this iteration (averaged over micro-steps)
            print(f"iter {i}: loss {accum_loss:.4f}, lr {lr:.2e}, time {dt*1000:.2f}ms")
            
        # Checkpointing
        if i > 0 and i % config.checkpoint_interval == 0:
            print(f"Saving checkpoint at iteration {i}...")
            checkpoint = {
                'iteration': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': config,
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, os.path.join(config.checkpoint_dir, f"ckpt_{i}.pt"))

    print("Training complete!")

if __name__ == "__main__":
    train()
