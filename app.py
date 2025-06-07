from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch
import numpy as np
import soundfile as sf
import os
import requests

app = Flask(__name__)
CORS(app)

# Load Sesame CSM-1B model with Metal support
device = torch.device("mps")  # Metal Performance Shaders (optimization for Mac OS)
sesame_model = torch.load("path/to/sesame-csm-1b.pth", map_location=device)
sesame_model.eval()

# Local storage for voice profiles
voice_profiles = {}

# Ollama LLM endpoint (local instance)
OLLAMA_URL = "http://localhost:11434/api/generate"

@app.route("/generate_audio", methods=["POST"])
def generate_audio():
    data = request.json
    text = data["text"]
    voice_id = data.get("voice_id", "default")
    
    # Generate audio using Sesame model (placeholder implementation)
    with torch.no_grad():
        audio = sesame_model.generate(text, voice_profiles.get(voice_id, None)).cpu().numpy()
    output_path = "output.wav"
    sf.write(output_path, audio, 22050)
    return jsonify({"audio_path": output_path})

@app.route("/clone_voice", methods=["POST"])
def clone_voice():
    audio_file = request.files["audio"]
    audio_data, sr = sf.read(audio_file)
    
    # Extract voice characteristics (placeholder)
    voice_profile = process_voice(audio_data)
    voice_id = str(len(voice_profiles))
    voice_profiles[voice_id] = voice_profile
    
    return jsonify({"voice_id": voice_id})

@app.route("/conversation", methods=["POST"])
def conversation():
    data = request.json
    user_input = data["input"]
    
    # Send input to Ollama LLM
    ollama_response = requests.post(OLLAMA_URL, json={"prompt": user_input, "model": "llama"}).json()
    response_text = ollama_response["response"]
    
    # Generate audio from LLM response
    with torch.no_grad():
        audio = sesame_model.generate(response_text, voice_profiles.get("default", None)).cpu().numpy()
    output_path = "conversation.wav"
    sf.write(output_path, audio, 22050)
    return jsonify({"audio_path": output_path})

@app.route("/download/<path:filename>")
def download(filename):
    return send_file(filename, as_attachment=True)

def process_voice(audio_data):
    # Placeholder for voice cloning logic
    return np.mean(audio_data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
