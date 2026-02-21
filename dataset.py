# MiniGPT dataset file
import torch
import tiktoken

class TextDataset:
    def __init__(self, text, block_size):
        self.block_size = block_size
        self.enc = tiktoken.get_encoding("gpt2")
        
        # Use BPE instead of character-level
        self.data = torch.tensor(self.enc.encode_ordinary(text), dtype=torch.long)
        self.vocab_size = self.enc.n_vocab

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data)-self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x, y
