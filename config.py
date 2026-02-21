# MiniGPT config file
import torch

block_size = 256 # Longer context for code
batch_size = 64
n_embd = 512 # Wider embeddings for more concepts
n_heads = 8
n_layers = 8 # Deeper model for better reasoning
dropout = 0.2
learning_rate = 3e-4
max_iters = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
