# Sesame Web App

A local web application for interacting with the Sesame Conversational Speech Model (CSM-1B), providing text-to-speech, voice cloning, and conversation mode with a local Ollama LLM instance. Optimized for macOS with M2 Max chip using MLX for Metal GPU acceleration.

## Features
- **Text-to-Speech**: Input text to generate human-like speech using Sesame CSM-1B.
- **Audio Preview**: Play generated audio with real-time waveform visualization.
- **Dynamic Visualization**: Audio playback visualized with a waveform (inspired by Sesame's giggly circle).
- **Download Audio**: Save generated audio files as WAV.
- **Voice Cloning**: Upload or record audio to create persistent voice profiles.
- **Conversation Mode**: Speak with the model, powered by a local Ollama LLM instance.
- **M2 Max Optimization**: Uses MLX framework for Apple Silicon GPU acceleration.

## Prerequisites
- **Hardware**: Mac with M2 Max chip.
- **Software**:
  - Python 3.12 (avoid 3.13 due to `sentencepiece` issues).
  - Node.js 18+.
  - Sesame CSM-1B model weights (from [senstella/csm-1b-mlx](https://huggingface.co/senstella/csm-1b-mlx)).
  - Ollama LLM instance (running locally on port 11434).
- **Dependencies**:
  - Python: `csm-mlx`, `numpy`, `audiofile`, `huggingface_hub`, `flask`, `flask-cors`, `requests`, `sentencepiece`.
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
   pip install git+https://github.com/senstella/csm-mlx --upgrade
   pip install numpy audiofile huggingface_hub flask flask-cors requests sentencepiece
   ```

3. **Download Sesame Model Weights**:
   - Create a `weights/` directory in the project:
     ```bash
     mkdir weights
     ```
   - Download the required files from Hugging Face:
     ```bash
     python -c "from huggingface_hub import hf_hub_download; \
     hf_hub_download(repo_id='senstella/csm-1b-mlx', filename='ckpt.safetensors', local_dir='weights'); \
     hf_hub_download(repo_id='senstella/csm-1b-mlx', filename='config.json', local_dir='weights'); \
     hf_hub_download(repo_id='senstella/csm-1b-mlx', filename='tokenizer.json', local_dir='weights'); \
     hf_hub_download(repo_id='senstella/csm-1b-mlx', filename='tokenizer_config.json', local_dir='weights'); \
     hf_hub_download(repo_id='senstella/csm-1b-mlx', filename='sentencepiece.bpe.model', local_dir='weights'); \
     hf_hub_download(repo_id='senstella/csm-1b-mlx', filename='generation_config.json', local_dir='weights')"
     ```
   - **Note**: If `sentencepiece.bpe.model` is not available in the repository, train a model using `sentencepiece`:
     ```bash
     echo -e "Hello from Sesame.\nThis is a test.\nHow are you doing today?" > corpus.txt
     python -c "import sentencepiece as spm; \
     spm.SentencePieceTrainer.train(input='corpus.txt', model_prefix='sentencepiece', vocab_size=1000, model_type='bpe', character_coverage=1.0); \
     import os; os.rename('sentencepiece.model', 'weights/sentencepiece.bpe.model')"
     ```
   - Alternatively, download manually from [senstella/csm-1b-mlx](https://huggingface.co/senstella/csm-1b-mlx) and place in `weights/`.
   - Ensure Hugging Face authentication is set up (see [Hugging Face Docs](https://huggingface.co/docs/hub/security-tokens)).

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
- `weights/`: Directory containing model weights and configuration files.

## Notes
- **Voice Cloning**: The `process_voice` function in `app.py` is a placeholder. Implement `csm-mlx`’s voice cloning logic (e.g., using `Segment` objects) as per the [csm-mlx documentation](https://github.com/senstella/csm-mlx).
- **Weights Storage**: Weights are stored in `weights/` within the project directory for simplicity. Alternatively, store them in a shared directory (e.g., `~/sesame-weights/`) and update `app.py` to use that path.
- **Performance**: Optimized for M2 Max with MLX, leveraging Metal for GPU acceleration.

## Troubleshooting
- **Model Loading Error**: Verify the weights path in `app.py` and ensure all required files are in `weights/`.
- **Tokenizer Error**: Ensure `sentencepiece.bpe.model` is in `weights/`. If missing, train a model as shown above.
- **Ollama Not Responding**: Check if Ollama is running on `http://localhost:11434`.
- **Audio Issues**: Ensure browser supports WAV playback and WaveSurfer.js is loaded.

## License
This project is licensed under the MIT License. The Sesame CSM-1B model is licensed under Apache 2.0 (see [senstella/csm-1b-mlx](https://huggingface.co/senstella/csm-1b-mlx)).

## Acknowledgments
- [Sesame AI](https://github.com/SesameAILabs) for the CSM-1B model.
- [senstella/csm-1b-mlx](https://github.com/senstella/csm-mlx) for MLX implementation.
- [Ollama](https://ollama.ai/) for local LLM hosting.
- [WaveSurfer.js](https://wavesurfer-js.org/) for audio visualization.
