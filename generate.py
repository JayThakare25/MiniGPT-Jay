import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from model import MiniGPT
from config import MiniGPTConfig
import os
import shutil
from gemini_utils import get_gemini_response

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

def generate_response(prompt, model, tokenizer, config, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    model.eval()
    # New Template matching the mixed training data
    formatted_prompt = f"Instruction: {prompt}\nThought: "
    idx = tokenizer.encode(formatted_prompt, return_tensors="pt").to(config.device)
    
    generated = idx
    for _ in range(max_new_tokens):
        idx_cond = generated if generated.size(1) <= config.block_size else generated[:, -config.block_size:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

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
    
    # Check if we should restore from Drive first
    if not os.path.exists(checkpoint_path) and os.path.exists(config.drive_checkpoint_dir):
        drive_latest = os.path.join(config.drive_checkpoint_dir, "latest.pt")
        if os.path.exists(drive_latest):
            print(f"Restoring latest checkpoint from Drive to local...")
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            shutil.copy2(drive_latest, checkpoint_path)

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    else:
        print("No checkpoint found. Running with randomly initialized weights.")

    print("-" * 30)
    print("MiniGPT Technical Assistant CLI")
    print("Modes: [1] Local MiniGPT  [2] Google Gemini (Free)")
    print("-" * 30)
    
    mode = input("Select mode (1 or 2): ")
    use_gemini = (mode == "2")

    while True:
        prompt = input(f"\n[{'Gemini' if use_gemini else 'Local'}] Enter question (or 'exit'): ")
        if prompt.lower() == 'exit':
            break
            
        print("\nGenerating response...")
        if use_gemini:
            response = get_gemini_response(prompt, config)
        else:
            response = generate_response(prompt, model, tokenizer, config)
            
        print(f"\nAI: {response}")

if __name__ == "__main__":
    main()
