#hello
import torch
from config import MiniGPTConfig
from model import MiniGPT
import sys

def test_model():
    print("Testing model instantiation and forward pass...")
    config = MiniGPTConfig(block_size=128, n_layer=2, n_head=4, n_embd=128) # Smaller for testing
    model = MiniGPT(config)
    
    # Dummy input
    idx = torch.randint(0, config.vocab_size, (1, 10))
    logits, loss = model(idx)
    
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (1, 1, config.vocab_size), f"Expected (1, 1, {config.vocab_size}), got {logits.shape}"
    print("Model test passed!\n")

def test_imports():
    print("Testing imports...")
    try:
        import transformers
        import datasets
        print("Dataset and Transformers imports successful!")
    except ImportError as e:
        print(f"Import failed: {e}. Note: This is expected if 'requirements.txt' is not installed locally.")

if __name__ == "__main__":
    test_model()
    test_imports()
