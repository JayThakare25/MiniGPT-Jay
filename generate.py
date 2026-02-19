# MiniGPT generation script
import torch
import torch.nn.functional as F
from model import MiniGPTModel
from dataset import TextDataset
import requests

# -------- CONFIG --------
block_size = 128
n_embd = 384
n_heads = 6
n_layers = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset (needed for tokenizer)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
data = TextDataset(text, block_size)

vocab_size = len(data.stoi)

# Create model
model = MiniGPTModel(
    vocab_size,
    n_embd,
    n_heads,
    n_layers,
    block_size
).to(device)


# -------- LOAD TRAINED WEIGHTS --------
model.load_state_dict(torch.load("minigpt_weights.pt", map_location=device))
model.eval()


# -------- GENERATION FUNCTION --------
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -block_size:]

        logits = model(idx_cond)
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return idx

# Start token (random character)
start = torch.zeros((1,1), dtype=torch.long).to(device)

out = generate(model, start, max_new_tokens=200)[0].tolist()

# Convert numbers back to characters
decoded = ''.join([data.itos[i] for i in out])
print(decoded)
