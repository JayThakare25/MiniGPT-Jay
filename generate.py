import torch
from transformers import GPT2Tokenizer
from model import MiniGPT
from config import MiniGPTConfig
import os

def generate_response(prompt, model, tokenizer, config, max_new_tokens=200):
    model.eval()
    # Format the prompt to match the training format
    formatted_prompt = f"Instruction: {prompt}\nInput: \nResponse: "
    
    idx = tokenizer.encode(formatted_prompt, return_tensors="pt").to(config.device)
    
    with torch.no_grad():
        completion = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.7, top_k=40)
        
    full_text = tokenizer.decode(completion[0], skip_special_tokens=True)
    # Extract only the response part
    response = full_text.split("Response:")[-1].strip()
    return response

def main():
    config = MiniGPTConfig()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    model = MiniGPT(config).to(config.device)
    checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint found. Running with randomly initialized weights (for structure testing).")

    print("-" * 30)
    print("MiniGPT Technical Assistant CLI")
    print("-" * 30)
    
    while True:
        prompt = input("\nEnter your technical question (or 'exit'): ")
        if prompt.lower() == 'exit':
            break
            
        print("\nGenerating response...")
        response = generate_response(prompt, model, tokenizer, config)
        print(f"\nAI: {response}")

if __name__ == "__main__":
    main()
