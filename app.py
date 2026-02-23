import streamlit as st
import torch
from transformers import GPT2Tokenizer
from model import MiniGPT
from config import MiniGPTConfig
import os
import shutil
from gemini_utils import get_gemini_response

# Page config
st.set_page_config(page_title="MiniGPT Technical Assistant", page_icon="🤖", layout="wide")

# Custom CSS for glassmorphism and premium look
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    .chat-bubble {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .ai-bubble {
        border-left: 5px solid #00d4ff;
    }
    .user-bubble {
        border-left: 5px solid #ff007a;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    config = MiniGPTConfig()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = MiniGPT(config).to(config.device)
    
    checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pt")
    
    # Auto-restore from Drive if local missing (Colab support)
    if not os.path.exists(checkpoint_path) and os.path.exists(config.drive_checkpoint_dir):
        drive_latest = os.path.join(config.drive_checkpoint_dir, "latest.pt")
        if os.path.exists(drive_latest):
            st.info("Restoring latest weights from Google Drive...")
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            shutil.copy2(drive_latest, checkpoint_path)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.success(f"Model loaded from iteration {checkpoint.get('iteration', 'unknown')}")
    else:
        st.warning("No checkpoint found. Model is using random weights.")
    
    model.eval()
    return model, tokenizer, config

def generate_response(prompt, model, tokenizer, config):
    # Match the new training template
    formatted_prompt = f"Instruction: {prompt}\nThought: "
    idx = tokenizer.encode(formatted_prompt, return_tensors="pt").to(config.device)
    
    with torch.no_grad():
        # Simple generation for speed
        generated = model.generate(idx, max_new_tokens=256, temperature=0.7, top_k=40)
    
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return full_text

# UI Layout
st.title("🤖 MiniGPT Technical Assistant")
st.markdown("---")

model, tokenizer, config = load_model()

# Sidebar configuration
st.sidebar.title("Configuration")
use_gemini = st.sidebar.toggle("Use Gemini (Free API)", value=False)
if use_gemini:
    st.sidebar.info("Using Google Gemini 1.5 Flash.")
else:
    st.sidebar.info("Using local MiniGPT model.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.container():
        role_class = "user-bubble" if message["role"] == "user" else "ai-bubble"
        st.markdown(f"""<div class="chat-bubble {role_class}">
            <b>{'You' if message['role'] == 'user' else 'AI'}:</b><br>{message['content']}
        </div>""", unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Ask a technical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        if use_gemini:
            response = get_gemini_response(prompt, config)
        else:
            response = generate_response(prompt, model, tokenizer, config)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()
