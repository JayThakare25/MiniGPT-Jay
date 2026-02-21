# MiniGPT config file
import torch

block_size = 256
batch_size = 32 # Lowered batch size for Free Tier GPU stability
n_embd = 384     # Balanced for T4 GPU memory
n_heads = 6
n_layers = 8    # Solid depth that trains quickly on Free Tier
dropout = 0.2
learning_rate = 6e-4
max_iters = 15000 # 120% Excellence Run
warmup_iters = 500
min_lr = 6e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
