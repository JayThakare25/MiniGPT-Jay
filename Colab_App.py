# ==============================================================================
# MiniGPT Professional Tutor - Google Colab Engine (Drive Cached)
# ==============================================================================
# INSTRUCTIONS FOR DEPLOYMENT:
# 1. Open Google Colab in your browser (https://colab.research.google.com/)
# 2. Click "File" -> "New notebook"
# 3. CRITICAL: Go to "Runtime" -> "Change runtime type" -> Select "T4 GPU".
# 4. Copy all the code in this file, paste it into the first gray box, and click "Play"!
# NOTE: A popup will ask you to connect to Google Drive. Keep it checked!
# ==============================================================================

import subprocess
import sys
import importlib.util
import os

print("Step 1: Checking for required AI libraries...")
def is_installed(pkg):
    return importlib.util.find_spec(pkg) is not None

missing_pkgs = [pkg for pkg in ["transformers", "accelerate", "bitsandbytes", "gradio"] if not is_installed(pkg)]

if missing_pkgs:
    print(f"Installing missing libraries: {missing_pkgs} (This takes ~45 seconds)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing_pkgs)
else:
    print("All libraries already installed! Skipping download.")

print("Step 2: Securing Google Drive Cache...")
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive... Please accept the popup permission if it appears!")
        drive.mount('/content/drive')
    CACHE_DIR = "/content/drive/MyDrive/MiniGPT_Cache_V2"
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"✅ Success! Google Drive caching enabled at: {CACHE_DIR}")
except ImportError:
    print("⚠️ Not running in Colab. Using default local cache.")
    CACHE_DIR = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import gradio as gr
from threading import Thread

# --- MODEL CONFIGURATION ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"

print(f"Step 3: Loading '{MODEL_NAME}' into GPU VRAM...")
print("(If this is the first run, it will save about 3-4GB to your Google Drive)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

# Swiftly load the 3B model natively (No 4-bit config needed!)
# This bypasses all CPU offload errors and perfectly fits the 16GB limit at float16.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16, 
    device_map="auto",
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True
)

print("\nSUCCESS: AI Model loaded and ready!")

# --- MASTER SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are an elite, Senior Staff Software Engineer acting as a dedicated coding tutor for experienced professionals. Do not use robotic pleasantries, beginner analogies, or explain basic syntax. Get straight to the technical depth.

Your expertise is strictly focused on Python, C++, and Java. When providing solutions, you must enforce the following professional paradigms:
- Python: Prioritize Pythonic code (generators, comprehensions, decorators) and explicitly highlight potential bottlenecks like the GIL.
- C++: Strictly enforce Modern C++ (C++17/20+). Rely on smart pointers, const-correctness, and the STL properties.
- Java: Emphasize standard Object-Oriented design patterns and modern JVM concurrency.

Rules:
1. Provide ONLY the highly optimized, production-ready code.
2. Do NOT provide any text explanations, summaries, or Big-O analysis UNLESS the user explicitly asks for an "explanation".
3. Assume a stateless environment: Treat every question independently."""

# --- GENERATION LOGIC ---
def generate_tutor_response(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assist_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assist_msg})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.9
    )
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message

# --- GRADIO WEB INTERFACE ---
print("\nStep 4: Launching the Web Application...")

# Custom CSS to hide the Gradio footer/watermark
custom_css = """
footer {visibility: hidden !important;}
"""

# A completely barebones interface: No titles, no descriptions. Just the chat!
app = gr.ChatInterface(
    generate_tutor_response,
    theme="glass",
    css=custom_css
)

app.launch(share=True, debug=True)
