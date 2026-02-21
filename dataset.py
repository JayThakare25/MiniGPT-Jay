# MiniGPT dataset file
import torch

class TextDataset:
    def __init__(self, text, block_size):
        self.block_size = block_size
        self.text = text

        # Simple character-level tokenizer
        self.chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}

        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data)-self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x, y
