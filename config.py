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

    learning_rate: float = 6e-4
    max_iters: int = 5000       # Total training steps
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0      # Clip gradients at this value
    
    # Checkpointing
    checkpoint_interval: int = 100

    checkpoint_dir: str = "checkpoints"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
