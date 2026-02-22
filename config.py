import torch
from dataclasses import dataclass

@dataclass
class MiniGPTConfig:
    # Model Architecture
    block_size: int = 512       # Context window size
    vocab_size: int = 50257     # GPT-2 vocab size
    n_layer: int = 12           # Number of transformer layers
    n_head: int = 12            # Number of attention heads
    n_embd: int = 768           # Embedding dimension
    dropout: float = 0.1
    bias: bool = False          # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # Training
    batch_size: int = 4         # Micro-batch size (reduced for T4 VRAM)
    gradient_accumulation_steps: int = 8 # Effective batch size = 4 * 8 = 32

    learning_rate: float = 3e-4
    max_iters: int = 30000       # Optimized for a ~8 hour overnight run
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0      # Clip gradients at this value
    target_loss: float = 0.35    # Safe "sweet spot" to prevent overfitting
    early_stop_threshold: int = 5 # Number of consecutive logs below target_loss to stop
    
    # Checkpointing

    checkpoint_interval: int = 1200   # Backup every 1,200 iters as requested
    latest_interval: int = 100        # Overwrite latest.pt for resilience

    checkpoint_dir: str = "checkpoints"
    drive_checkpoint_dir: str = "/content/drive/MyDrive/MiniGPT-Jay/checkpoints" # Optional drive sync

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
