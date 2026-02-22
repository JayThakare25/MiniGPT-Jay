import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np

class TechnicalDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the dataset (Code Alpaca for technical/coding instructions)
        # Using a small subset or streaming for efficiency if needed
        dataset = load_dataset("lucasmansilla/code-alpaca-20k", split=split)
        
        self.data = dataset
        self.block_size = config.block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format: Instruction + Input + Output
        text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nResponse: {item['output']}<|endoftext|>"
        
        tokens = self.tokenizer.encode(
            text, 
            max_length=self.block_size + 1, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        ).squeeze(0)
        
        x = tokens[:-1]
        y = tokens[1:]
        
        return x, y

def get_dataloader(config, split="train"):
    dataset = TechnicalDataset(config, split=split)
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=(split=="train"), 
        pin_memory=True,
        num_workers=2
    )
