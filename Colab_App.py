# ==============================================================================
# OptiCode Backend Engine
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
import re
import io
import logging

# Suppress HuggingFace authentication warnings but keep progress bars
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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

# --- OPTICODE CONFIGURATION ---
_INTERNAL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"

print(f"Step 3: Booting up OptiCode-3B Engine into GPU VRAM...")
print("(Initializing model architecture and allocating memory...)")

tokenizer = AutoTokenizer.from_pretrained(_INTERNAL_ID, cache_dir=CACHE_DIR)

# Swiftly load the 3B model natively (No 4-bit config needed!)
# This bypasses all CPU offload errors and perfectly fits the 16GB limit at float16.
model = AutoModelForCausalLM.from_pretrained(
    _INTERNAL_ID,
    torch_dtype=torch.float16, 
    device_map="auto",
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True
)

print("\nSUCCESS: AI Model loaded and ready!")

# --- MASTER SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are OptiCode, an advanced AI code engine.
If the user asks for code, you MUST output ONLY the markdown code block. Do NOT include ANY greetings, explanations, or conversational filler before or after the code.
If the user asks a general question or says hello, answer normally without using code blocks."""

VOICE_PROMPT = """You are OptiCode (aka Jarvis), an advanced AI conversational coding partner.
You are currently interacting with the user via VOICE. You must act natural, human-like, and conversational. 
Keep your verbal responses concise and natural. If the user asks for code, provide the code block as usual, and briefly explain it like a human partner would."""

# --- GENERATION LOGIC ---
def generate_code_response(message, history, execute_code=False):
    if message.startswith("__EXECUTE_ONLY__"):
        code = message.replace("__EXECUTE_ONLY__", "").strip()
        output_log = "\n\n### ⚡ Execution Output:\n```text\n"
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        try:
            exec(code, globals())
            output_log += redirected_output.getvalue()
        except Exception as e:
            output_log += f"Error: {e}\n"
        finally:
            sys.stdout = old_stdout
        output_log += "```"
        yield output_log
        return

    is_voice = False
    if message.startswith("[VOICE_MODE]"):
        is_voice = True
        message = message.replace("[VOICE_MODE]", "").strip()

    active_prompt = VOICE_PROMPT if is_voice else SYSTEM_PROMPT
    messages = [{"role": "system", "content": active_prompt}]
    for user_msg, assist_msg in history:
        # Strip internal flags from history so it doesn't confuse the model
        user_msg = user_msg.replace("[VOICE_MODE]", "").strip()
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assist_msg})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        **model_inputs,
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

    # Code Execution Agent Logic
    code_blocks = re.findall(r'```python\n(.*?)\n```', partial_message, re.DOTALL)
    if execute_code and code_blocks:
        output_log = "\n\n### ⚡ Execution Output:\n```text\n"
        has_error = False
        error_msg = ""
        for code in code_blocks:
            old_stdout = sys.stdout
            redirected_output = sys.stdout = io.StringIO()
            try:
                exec(code, globals())
                output_log += redirected_output.getvalue()
            except Exception as e:
                has_error = True
                error_msg = str(e)
                output_log += f"Error: {e}\n"
            finally:
                sys.stdout = old_stdout
        output_log += "```"
        partial_message += output_log
        yield partial_message

        if has_error:
            yield partial_message + "\n\n🤖 **Self-Healing Initiated... fixing error:**\n"
            repair_prompt = f"The code you just wrote threw this error:\n```text\n{error_msg}\n```\nPlease rewrite the code to fix this exact error. Provide ONLY the corrected code."
            repair_messages = messages + [{"role": "assistant", "content": partial_message}, {"role": "user", "content": repair_prompt}]
            repair_text = tokenizer.apply_chat_template(repair_messages, tokenize=False, add_generation_prompt=True)
            repair_inputs = tokenizer([repair_text], return_tensors="pt").to(model.device)
            streamer2 = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
            
            t2 = Thread(target=model.generate, kwargs=dict(**repair_inputs, streamer=streamer2, max_new_tokens=1024, temperature=0.3, top_p=0.9))
            t2.start()
            
            fix_message = ""
            for new_token in streamer2:
                fix_message += new_token
                yield partial_message + "\n\n🤖 **Self-Healing Initiated... fixing error:**\n\n" + fix_message

# --- GRADIO WEB INTERFACE ---
print("\nStep 4: Launching the Web Application...")

# Custom CSS to hide the Gradio footer/watermark
custom_css = """
footer {visibility: hidden !important;}
"""

# A completely barebones interface: No titles, no descriptions. Just the chat!
app = gr.ChatInterface(
    generate_code_response,
    additional_inputs=[gr.Checkbox(label="Run Python Code", value=False)],
    theme="glass",
    css=custom_css
)

app.launch(share=True, debug=True)
