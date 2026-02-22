import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from model import MiniGPT
from config import MiniGPTConfig
import os

def top_p_sampling(logits, p=0.9):
    """ Nucleus sampling: sample from the smallest set of tokens whose cumulative probability exceeds p. """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def generate_response(prompt, model, tokenizer, config, max_new_tokens=256, temperature=0.7, top_p=0.9):
    model.eval()
    formatted_prompt = f"Instruction: {prompt}\nInput: \nResponse: "
    idx = tokenizer.encode(formatted_prompt, return_tensors="pt").to(config.device)
    
    generated = idx
    for _ in range(max_new_tokens):
        idx_cond = generated if generated.size(1) <= config.block_size else generated[:, -config.block_size:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        
        if top_p is not None and top_p < 1.0:
            idx_next = top_p_sampling(logits, p=top_p)
        else:
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
        generated = torch.cat((generated, idx_next), dim=1)
        
        if idx_next.item() == tokenizer.eos_token_id:
            break
            
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
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
        print("No checkpoint found. Running with randomly initialized weights.")

    print("-" * 30)
    print("MiniGPT Technical Assistant CLI (Nucleus Sampling Enabled)")
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
