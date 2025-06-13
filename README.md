# Open Sesame - Voice Generation App
<img width="650" alt="Screenshot 2025-06-13 at 4 10 37 PM" src="https://github.com/user-attachments/assets/4badc795-f1eb-4c31-b168-5f931e5f330f" />


Open Sesame is a Streamlit-based web application that generates natural-sounding speech using the CSM (Conditional Speech Model). The application allows users to generate speech in different personas' voices by providing text input.

## Features

- 🎙️ Generate natural-sounding speech from text input
- 👤 Multiple persona support with different voice characteristics
- 💬 Conversation history tracking
- 🎧 Audio playback and download capabilities
- 🖥️ User-friendly web interface built with Streamlit

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- PyTorch
- Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/open_sesame.git
cd open_sesame
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Select a persona from the dropdown menu
4. Enter the text you want to convert to speech
5. Click "Generate Speech" to create the audio
6. Use the audio player to listen to the generated speech
7. Download the generated audio file if desired

## Persona System

The application supports multiple personas, each with their own voice characteristics. Personas are stored in the `personas` directory, with each persona having:
- A `transcript.json` file containing sample text and corresponding audio files
- WAV audio files for voice samples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

- CSM (Conditional Speech Model) for voice generation
- Streamlit for the web interface
- PyTorch and torchaudio for audio processing
