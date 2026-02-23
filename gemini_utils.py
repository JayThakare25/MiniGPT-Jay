import os
import google.generativeai as genai
from dotenv import load_dotenv

def get_gemini_response(prompt, config):
    """
    Fetches a response from Google's Gemini API (Free Tier).
    """
    # Load .env file
    load_dotenv()
    
    # Priority: 1. .env file, 2. Colab Secrets (userdata), 3. config.py
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        try:
            from google.colab import userdata
            # This can raise SecretNotFoundError if the key is missing
            # or NotebookAccessError if not granted access
            api_key = userdata.get('GEMINI_API_KEY')
        except Exception:
            # Fallback to config if Colab secret fails for ANY reason
            api_key = config.gemini_api_key
    
    if not api_key:
        return "Error: Gemini API Key not found. Please set GEMINI_API_KEY in your .env file or Colab Secrets."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config.gemini_model)
        
        # Consistent with technical assistant persona
        response = model.generate_content(
            f"You are a helpful technical assistant. {prompt}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=512
            )
        )
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"
