import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np

class TechnicalDataset(IterableDataset):
    def __init__(self, config, split="train"):
        super().__init__()
        self.config = config
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Using a stable version of Code Alpaca
        self.dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=split, streaming=True)
        self.block_size = config.block_size
        self.tokenizer_eos_id = self.tokenizer.eos_token_id

    def __iter__(self):
        buffer = []
        for item in self.dataset:
            # Format: Instruction + Input + Output
            text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nResponse: {item['output']}{self.tokenizer.eos_token}"
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            
            # Efficient Token Packing: Yield chunks of block_size + 1
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
                buffer = buffer[self.block_size:] # Consume block_size tokens


def get_dataloader(config, split="train"):
    dataset = TechnicalDataset(config, split=split)
    # IterableDataset doesn't support shuffle in DataLoader, usually handled in dataset or via buffer
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        pin_memory=True,
        num_workers=1
    )
