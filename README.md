# Sesame Web App

A local web application for interacting with the Sesame Conversational Speech Model (CSM-1B), providing text-to-speech, voice cloning, and conversation mode with a local Ollama LLM instance. Optimized for macOS with M2 Max chip using Metal GPU acceleration.

## Features
- **Text-to-Speech**: Input text to generate human-like speech using Sesame CSM-1B.
- **Audio Preview**: Play generated audio with real-time waveform visualization.
- **Dynamic Visualization**: Audio playback visualized with a waveform (inspired by Sesame's giggly circle).
- **Download Audio**: Save generated audio files as WAV.
- **Voice Cloning**: Upload or record audio to create persistent voice profiles.
- **Conversation Mode**: Speak with the model, powered by a local Ollama LLM instance.
- **M2 Max Optimization**: Leverages Metal Performance Shaders (MPS) for GPU acceleration.

## Prerequisites
- **Hardware**: Mac with M1 and above
- **Software**:
  - Python 3.10+
  - Node.js 18+
  - Sesame CSM-1B model (download from [Sesame GitHub](https://github.com/SesameAILabs/csm)).
  - Ollama LLM instance (running locally on port 11434).
- **Dependencies**:
  - Python: `torch`, `numpy`, `scipy`, `soundfile`, `flask`, `flask-cors`, `requests`.
  - JavaScript: React, WaveSurfer.js, Tailwind CSS (via CDN).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd sesame-web-app
   ```

2. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch numpy scipy soundfile flask flask-cors requests
   ```

3. **Download Sesame Model**:
   - Download the CSM-1B model from [Sesame GitHub](https://github.com/SesameAILabs/csm).
   - Update the model path in `app.py` (replace `"path/to/sesame-csm-1b.pth"`).

4. **Set Up Ollama**:
   - Install and run Ollama locally: [Ollama Documentation](https://ollama.ai/).
   - Ensure it’s running on `http://localhost:11434`.

5. **Prepare Front-End**:
   - The front-end uses CDN-hosted libraries (React, WaveSurfer.js, Tailwind CSS).
   - No additional installation is required for JavaScript dependencies.

## Running the App

1. **Start the Flask Back-End**:
   ```bash
   python app.py
   ```
   The server runs on `http://localhost:5000`.

2. **Run the Front-End**:
   - Open `index.html` in a browser, or serve it with a static server:
     ```bash
     npx serve
     ```
   - Access the app at `http://localhost:3000` (or the port provided by `serve`).

## Usage
- **Text-to-Speech**: Enter text in the textarea and click "Generate Audio" to create speech.
- **Audio Playback**: Audio plays automatically with a waveform visualization.
- **Download**: Click "Download" to save the generated WAV file.
- **Voice Cloning**: Upload an audio file to create a new voice profile, selectable from the dropdown.
- **Conversation Mode**: Click "Start Conversation" to interact with the Ollama LLM, with responses spoken by Sesame.
- **Voice Profiles**: Persist in memory (extend to localStorage or SQLite for full persistence).

## Project Structure
- `app.py`: Flask back-end handling TTS, voice cloning, and LLM integration.
- `index.html`: React front-end with UI for all features.
- `README.md`: This file.

## Notes
- **Voice Cloning**: The `process_voice` function in `app.py` is a placeholder. Implement Sesame’s voice cloning logic as per the model’s documentation.
- **Persistence**: Voice profiles are stored in memory. For persistent storage, integrate localStorage or a SQLite database.
- **Performance**: Optimized for M2 Max with PyTorch’s `mps` backend. Ensure CUDA is not used, as it’s incompatible with macOS Metal.
- **Ollama**: Ensure the LLM model (e.g., Llama) is pulled and running in Ollama.

## Troubleshooting
- **Model Loading Error**: Verify the Sesame model path in `app.py`.
- **Ollama Not Responding**: Check if Ollama is running on `http://localhost:11434`.
- **Audio Issues**: Ensure browser supports WAV playback and WaveSurfer.js is loaded.

## License
This project is licensed under the MIT License. The Sesame CSM-1B model is licensed under Apache 2.0 (see [Sesame GitHub](https://github.com/SesameAILabs/csm)).

## Acknowledgments
- [Sesame AI](https://github.com/SesameAILabs) for the CSM-1B model.
- [Ollama](https://ollama.ai/) for local LLM hosting.
- [WaveSurfer.js](https://wavesurfer-js.org/) for audio visualization.
