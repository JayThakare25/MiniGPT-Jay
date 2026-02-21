# MiniGPT config file
import torch

block_size = 128
batch_size = 64
n_embd = 384
n_heads = 6
n_layers = 6
learning_rate = 3e-4
max_iters = 4000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
