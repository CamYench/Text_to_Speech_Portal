import os
import torch
import torchaudio
import streamlit as st
from pathlib import Path
from generator import Generator, Segment, load_csm_1b
import tempfile
import json
import warnings
import logging

# Suppress various warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Suppress Streamlit's file watcher warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def load_persona(persona_dir):
    """Load a persona's context from a directory containing wav files and a transcript.json"""
    if persona_dir == "default":
        return []
        
    persona_path = Path(persona_dir)
    transcript_path = persona_path / "transcript.json"
    
    if not transcript_path.exists():
        st.warning(f"No transcript.json found in {persona_dir}")
        return []
    
    try:
        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in {transcript_path}")
        return []
    
    # Only use these 20 samples for context
    selected_files = [
        "Sample1.wav",
        "Sample2.wav",
        "Sample3.wav",
        "Sample4.wav",
        "Sample5.wav",
        "Sample6.wav",
        "Sample7.wav",
        "Sample8.wav",
        "Sample9.wav",
        "Sample10.wav",
        "Sample11.wav",
        "Sample12.wav",
        "Sample13.wav",
        "Sample14.wav",
        "Sample15.wav",
        "Sample16.wav",
        "Sample17.wav",
        "Sample18.wav",
        "Sample19.wav",
        "Sample20.wav",
    ]
    
    segments = []
    for item in transcript_data:
        if item['audio_file'] not in selected_files:
            continue
        audio_path = persona_path / item['audio_file']
        if not audio_path.exists():
            st.warning(f"Audio file {item['audio_file']} not found in {persona_dir}")
            continue
        try:
            audio_tensor, sample_rate = torchaudio.load(str(audio_path))
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.squeeze(0), 
                orig_freq=sample_rate, 
                new_freq=24000  # CSM model's sample rate
            )
            segment = Segment(
                text=item['text'],
                speaker=0,  # All segments use speaker 0
                audio=audio_tensor
            )
            segments.append(segment)
        except Exception as e:
            st.error(f"Error loading audio file {item['audio_file']}: {str(e)}")
            continue
    
    return segments

def get_available_personas():
    """Get list of available personas from the personas directory"""
    personas_dir = Path("personas")
    personas = ["default"]  # Add default option first
    if personas_dir.exists():
        personas.extend([d.name for d in personas_dir.iterdir() if d.is_dir()])
    return personas

def save_audio(audio_tensor, filename):
    """Save audio tensor to a temporary file and return the file path"""
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    torchaudio.save(file_path, audio_tensor.unsqueeze(0).cpu(), 24000)
    return file_path

# Initialize the model
@st.cache_resource
def load_model():
    try:
        return load_csm_1b(device="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Streamlit UI
st.title("CSM Voice Generator")

# Load model
with st.spinner("Loading model..."):
    generator = load_model()
    if generator is None:
        st.error("Failed to load model. Please check the console for details.")
        st.stop()

# Persona selection
personas = get_available_personas()
selected_persona = st.selectbox("Select Persona", personas)

# Load selected persona's context
context = load_persona(f"personas/{selected_persona}" if selected_persona != "default" else "default")

# Display conversation history
st.subheader("Conversation History")
for i, segment in enumerate(st.session_state.conversation_history):
    st.write(f"{segment.text}")
    if hasattr(segment, 'audio_path'):
        st.audio(segment.audio_path)

# Input for new text
new_text = st.text_input("Enter text to generate speech:")

if st.button("Generate Speech"):
    if new_text:
        with st.spinner("Generating speech..."):
            try:
                # Generate audio
                audio = generator.generate(
                    text=new_text,
                    speaker=0,  # Always use speaker 0
                    context=context + st.session_state.conversation_history,
                    max_audio_length_ms=10_000,
                )
                
                # Save audio to temporary file
                audio_path = save_audio(audio, f"generated_{len(st.session_state.conversation_history)}.wav")
                
                # Create segment for history
                segment = Segment(
                    text=new_text,
                    speaker=0,
                    audio=audio
                )
                segment.audio_path = audio_path
                
                # Add to conversation history
                st.session_state.conversation_history.append(segment)
                
                # Display the new audio
                st.audio(audio_path)
                
                # Add download button
                with open(audio_path, 'rb') as f:
                    st.download_button(
                        label="Download Audio",
                        data=f,
                        file_name=f"generated_{len(st.session_state.conversation_history)}.wav",
                        mime="audio/wav"
                    )
            except Exception as e:
                st.error(f"Error generating speech: {str(e)}")

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []
    st.experimental_rerun() 