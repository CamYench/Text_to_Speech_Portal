from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from csm_mlx import CSM, csm_1b, generate
import numpy as np
import audiofile
import os
import requests

app = Flask(__name__)
CORS(app)

# Initialize Sesame CSM-1B model with MLX
csm = CSM(csm_1b())
weight_path = "weights/model.safetensors"  # Path to weights in project directory
csm.load_weights(weight_path)

# Local storage for voice profiles
voice_profiles = {}

# Ollama LLM endpoint (local instance)
OLLAMA_URL = "http://localhost:11434/api/generate"

@app.route("/generate_audio", methods=["POST"])
def generate_audio():
    data = request.json
    text = data["text"]
    voice_id = data.get("voice_id", "default")
    
    # Generate audio using Sesame model
    audio = generate(
        csm,
        text=text,
        speaker=0,  # Adjust based on voice_profiles
        context=voice_profiles.get(voice_id, []),
        max_audio_length_ms=10000,
        sampler={"temp": 0.8, "top_k": 50}
    )
    output_path = "output.wav"
    audiofile.write(output_path, np.array(audio), 24000)
    return jsonify({"audio_path": output_path})

@app.route("/clone_voice", methods=["POST"])
def clone_voice():
    audio_file = request.files["audio"]
    audio_data, sr = audiofile.read(audio_file)
    
    # Extract voice characteristics (placeholder for csm-mlx voice cloning)
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
    audio = generate(
        csm,
        text=response_text,
        speaker=0,
        context=voice_profiles.get("default", []),
        max_audio_length_ms=10000,
        sampler={"temp": 0.8, "top_k": 50}
    )
    output_path = "conversation.wav"
    audiofile.write(output_path, np.array(audio), 24000)
    return jsonify({"audio_path": output_path})

@app.route("/download/<path:filename>")
def download(filename):
    return send_file(filename, as_attachment=True)

def process_voice(audio_data):
    # Placeholder for voice cloning logic using csm-mlx
    return []  # Replace with actual context extraction

if __name__ == "__main__":
    app.run(debug=True, port=5000)
