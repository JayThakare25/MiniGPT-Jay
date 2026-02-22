import os
import torch
import random
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np

class TechnicalDataset(IterableDataset):
    def __init__(self, config, split="train"):
        super().__init__()
        self.config = config
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.model_max_length = 1e9 
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 1. Technical/Code Dataset (40% weight)
        self.code_ds = load_dataset("sahil2801/CodeAlpaca-20k", split=split, streaming=True)
        
        # 2. Conversational/Reasoning Dataset (30% weight)
        self.chat_ds = load_dataset("Open-Orca/OpenOrca", split=split, streaming=True)

        # 3. High-Complexity Technical Reasoning (30% weight)
        # WizardLM Evol-Instruct provides deeper 'why' and technical depth
        self.wizard_ds = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train", streaming=True)
        
        self.block_size = config.block_size
        self.tokenizer_eos_id = self.tokenizer.eos_token_id

    def format_item(self, item, is_chat=False):
        """ Formats data into a 'Thought -> Code -> Explanation' style. """
        if is_chat:
            # From OpenOrca
            instruction = item.get('instruction', '') or item.get('system_prompt', '')
            user_query = item.get('question', '')
            response = item.get('response', '')
            text = f"Instruction: {instruction}\nThought: Analyzing query: {user_query}\nResponse: {response}{self.tokenizer.eos_token}"
        else:
            # From CodeAlpaca
            instr = item['instruction']
            inp = item['input']
            out = item['output']
            # We inject a 'Thought' and 'Explanation' simulation for training
            text = f"Instruction: {instr}\nInput: {inp}\nThought: I will solve this by writing efficient code.\nResponse: {out}\nExplanation: This code implements the requested logic by handling the provided inputs.{self.tokenizer.eos_token}"
        
        return self.tokenizer.encode(text)

    def __iter__(self):
        buffer = []
        code_iter = iter(self.code_ds)
        chat_iter = iter(self.chat_ds)
        wiz_iter = iter(self.wizard_ds)
        
        while True:
            try:
                # 40% Code, 30% Chat, 30% High-Depth Theory
                r = random.random()
                if r < 0.4:
                    try: item = next(code_iter); is_chat = False
                    except StopIteration: code_iter = iter(self.code_ds); item = next(code_iter); is_chat = False
                elif r < 0.7:
                    try: item = next(chat_iter); is_chat = True
                    except StopIteration: chat_iter = iter(self.chat_ds); item = next(chat_iter); is_chat = True
                else:
                    try: item = next(wiz_iter); is_chat = True
                    except StopIteration: wiz_iter = iter(self.wizard_ds); item = next(wiz_iter); is_chat = True
                
                tokens = self.format_item(item, is_chat)
                buffer.extend(tokens)
                
                # Yield chunks
                while len(buffer) >= self.block_size + 1:
                    chunk = buffer[:self.block_size + 1]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y
                    buffer = buffer[self.block_size:]
            except Exception as e:
                print(f"Dataset iteration error: {e}")
                break

def get_dataloader(config, split="train"):
    dataset = TechnicalDataset(config, split=split)
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        pin_memory=True,
        num_workers=1
    )
