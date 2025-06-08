import streamlit as st
from csm_mlx import CSM, csm_1b, generate, Segment, load_adapters
from mlx_lm.sample_utils import make_sampler
from mlx import nn
import audiofile
import numpy as np
import os
import requests
import mlx.core as mx
import subprocess
from pathlib import Path
import json
import logging
import shutil

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize and quantize Sesame CSM-1B model with MLX
@st.cache_resource
def load_model(profile_dir=None):
    try:
        csm = CSM(csm_1b())
        if hasattr(mx, 'is_metal_available') and mx.is_metal_available():
            logging.info("M2 Max GPU (MPS) detected. Using default device for generation.")
        else:
            logging.warning("MPS not available or not supported in this mlx version. Falling back to CPU.")
        
        if profile_dir:
            try:
                # Check if profile directory exists
                if not os.path.exists(profile_dir):
                    raise FileNotFoundError(f"Profile directory not found: {profile_dir}")
                
                # Check for required files
                adapters_path = os.path.join(profile_dir, "adapters.safetensors")
                latest_path = os.path.join(profile_dir, "latest.safetensors")
                config_path = os.path.join(profile_dir, "profile_config.json")
                
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Profile config not found at {config_path}")
                
                # Try to load adapters
                if os.path.exists(adapters_path):
                    csm.load_weights(adapters_path)
                    load_adapters(csm, profile_dir)
                    logging.info(f"Loaded fine-tuned model from {profile_dir}")
                elif os.path.exists(latest_path):
                    csm.load_weights(latest_path)
                    load_adapters(csm, profile_dir)
                    logging.info(f"Loaded latest checkpoint from {profile_dir}")
                else:
                    raise FileNotFoundError(f"No model weights found in {profile_dir}")
                
            except Exception as e:
                logging.error(f"Failed to load fine-tuned model: {e}")
                st.error(f"Error loading fine-tuned model: {e}")
                # Fall back to base model
                logging.info("Falling back to base model")
                csm.load_weights("weights/ckpt.safetensors")
        else:
            csm.load_weights("weights/ckpt.safetensors")
            logging.info("Loaded base model")
        
        nn.quantize(csm)
        return csm
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        st.error(f"Failed to initialize model: {e}")
        return None

csm = load_model()

# Local storage for voice profiles
if 'voice_profiles' not in st.session_state:
    st.session_state.voice_profiles = {}
if 'fine_tuned_profiles' not in st.session_state:
    st.session_state.fine_tuned_profiles = []  # List of dicts with voice_id and path

# Load existing fine-tuned profiles
profiles_dir = "finetuned_weights"
if os.path.exists(profiles_dir):
    # Look for all profile configs in the directory
    for root, dirs, files in os.walk(profiles_dir):
        for file in files:
            if file == "profile_config.json":
                config_path = os.path.join(root, file)
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        # Only add if not already present (by voice_id and path)
                        if not any(
                            p.get("voice_id") == config.get("voice_id") and p.get("path") == config.get("path")
                            for p in st.session_state.fine_tuned_profiles
                        ):
                            st.session_state.fine_tuned_profiles.append(config)
                except Exception as e:
                    logging.error(f"Error loading profile config {config_path}: {e}")

OLLAMA_URL = "http://localhost:11434/api/generate"
CUSTOM_PROMPT = (
    "You are an AI, named Gemma, designed to generate plain text responses suitable for text-to-speech conversion."
    "Provide responses that are concise sentences without special formatting. Only include the text to be spoken."
)

st.title("Sesame Web App")

# Text input and audio generation
text = st.text_area("Enter text to convert to speech", "")
voice_options = list(st.session_state.voice_profiles.keys()) + ["default"] + [f"Fine-tuned {p['voice_id']}" for p in st.session_state.fine_tuned_profiles]
voice_id = st.selectbox("Select Voice", voice_options, index=voice_options.index("default") if "default" in voice_options else 0)
if st.button("Generate Audio"):
    sampler = make_sampler(temp=0.8, top_k=50)
    profile = next((p for p in st.session_state.fine_tuned_profiles if f"Fine-tuned {p['voice_id']}" == voice_id), None)
    profile_dir = profile["path"] if profile else None
    csm = load_model(profile_dir)
    try:
        audio = generate(
            csm, text, speaker=0, context=st.session_state.voice_profiles.get(voice_id, []), max_audio_length_ms=10000, sampler=sampler
        )
        output_path = "output.wav"
        audiofile.write(output_path, np.array(audio), 24000)
        st.session_state['audio_path'] = output_path
        st.audio(output_path, autoplay=True)
        st.download_button("Download Audio", output_path, file_name="output.wav")
    except ValueError as e:
        st.error(f"Generation error: Invalid padding sizes. Details: {e}")
        logging.error(f"Padding error: {e}")
    except Exception as e:
        st.error(f"Generation error: {e}")
        logging.error(f"General error: {e}")

# Voice cloning and fine-tuning
uploaded_file = st.file_uploader("Clone Voice (upload audio)", type=["wav"])
if uploaded_file:
    # Get the filename without extension as the voice ID
    voice_id = os.path.splitext(uploaded_file.name)[0]
    
    # Create a unique directory for this voice
    voice_dir = os.path.join("finetuned_weights", voice_id)
    os.makedirs(voice_dir, exist_ok=True)
    
    temp_path = os.path.join(os.getcwd(), "temp.wav")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    if st.button("Clone Voice"):
        try:
            audio_data, sr = audiofile.read(temp_path)
            audio_array = mx.array(audio_data)
            voice_profile = Segment(speaker=0, text="Cloned voice", audio=audio_array)
            st.session_state.voice_profiles[voice_id] = [voice_profile]
            st.success(f"Cloned voice with ID: {voice_id}")
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
            logging.error(f"Error processing audio file: {e}")
    
    if st.button("Fine-Tune and Save Profile"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Data Preparation
            status_text.text("Step 1/3: Preparing audio data...")
            progress_bar.progress(10)
            
            try:
                # Set environment variables for the subprocess
                env = os.environ.copy()
                env["TOKENIZERS_PARALLELISM"] = "false"
                
                dataset_path = subprocess.check_output(
                    ["python", "data_prep.py", temp_path],
                    stderr=subprocess.STDOUT,
                    env=env
                ).decode().strip()
                logging.info(f"Data preparation completed. Dataset saved at: {dataset_path}")
            except subprocess.CalledProcessError as e:
                error_msg = e.output.decode()
                logging.error(f"Data preparation failed: {error_msg}")
                raise ValueError(f"Data preparation failed: {error_msg}")
            
            progress_bar.progress(30)
            
            # Step 2: Fine-tuning
            status_text.text("Step 2/3: Fine-tuning model (this may take a while)...")
            try:
                # Check available disk space
                total, used, free = shutil.disk_usage(os.getcwd())
                free_mb = free // (1024 * 1024)
                if free_mb < 100:  # Require at least 100MB free
                    raise RuntimeError(f"Not enough disk space. Required: 100MB, Available: {free_mb}MB")
                
                output = subprocess.check_output(
                    ["python", "finetune_script.py", dataset_path, "--output-dir", voice_dir, "--voice-id", voice_id],
                    stderr=subprocess.STDOUT,
                    env=env
                ).decode().strip()
                logging.info("Fine-tuning completed successfully")
            except subprocess.CalledProcessError as e:
                error_msg = e.output.decode()
                logging.error(f"Fine-tuning failed: {error_msg}")
                if "Unable to write" in error_msg:
                    st.error("Fine-tuning failed: Not enough disk space. Please free up some space and try again.")
                else:
                    st.error(f"Fine-tuning failed: {error_msg}")
                raise ValueError(f"Fine-tuning failed: {error_msg}")
            except RuntimeError as e:
                st.error(str(e))
                raise
            
            progress_bar.progress(80)
            
            # Step 3: Process output and save profile
            status_text.text("Step 3/3: Saving voice profile...")
            
            # Verify and load profile config
            config_path = os.path.join(voice_dir, "profile_config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Profile config not found at {config_path}")
            
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Check if profile already exists
            if any(
                p.get("voice_id") == config.get("voice_id") and p.get("path") == config.get("path")
                for p in st.session_state.fine_tuned_profiles
            ):
                st.warning(f"Profile {config['voice_id']} already exists. Skipping...")
            else:
                st.session_state.fine_tuned_profiles.append(config)
                st.success(f"Profile saved as Fine-tuned {config['voice_id']}")
            
            progress_bar.progress(100)
            status_text.text("Voice cloning completed successfully!")
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error during voice cloning: {error_msg}")
            logging.error(f"Voice cloning error details: {error_msg}")
            status_text.text("Voice cloning failed!")
        finally:
            # Clean up temporary files
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if 'dataset_path' in locals() and os.path.exists(dataset_path):
                try:
                    os.remove(dataset_path)
                except Exception as e:
                    logging.warning(f"Failed to remove temporary dataset file: {e}")

# Conversation mode
if 'conversation_mode' not in st.session_state:
    st.session_state.conversation_mode = False
if st.button("Start/Stop Conversation"):
    st.session_state.conversation_mode = not st.session_state.conversation_mode
if st.session_state.conversation_mode:
    user_input = st.text_input("Conversation Input")
    if st.button("Send"):
        try:
            response = requests.post(OLLAMA_URL, json={"prompt": f"{CUSTOM_PROMPT}\n\n{user_input}", "model": "gemma3:4b", "stream": False}, timeout=10)
            response.raise_for_status()
            ollama_response = response.json()
            if "response" in ollama_response:
                response_text = ollama_response["response"]
            else:
                st.error("Unexpected response format.")
                response_text = None
            if response_text:
                sampler = make_sampler(temp=0.8, top_k=50)
                profile = next((p for p in st.session_state.fine_tuned_profiles if f"Fine-tuned {p['voice_id']}" == voice_id), None)
                profile_dir = profile["path"] if profile else None
                csm = load_model(profile_dir)
                audio = generate(
                    csm, response_text, speaker=0, context=st.session_state.voice_profiles.get("default", []), max_audio_length_ms=10000, sampler=sampler
                )
                output_path = "conversation.wav"
                audiofile.write(output_path, np.array(audio), 24000)
                st.session_state['audio_path'] = output_path
                st.audio(output_path, autoplay=True)
                st.download_button("Download Conversation", output_path, file_name="conversation.wav")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to Ollama API: {e}")
        except ValueError as e:
            st.error(f"Generation error: Invalid padding sizes. Details: {e}")
            logging.error(f"Padding error: {e}")
        except Exception as e:
            st.error(f"Conversation error: {e}")
            logging.error(f"General error: {e}")

def generate(csm, text, speaker=0, context=[], max_audio_length_ms=10000, sampler=None):
    segment_tokens, segment_tokens_mask = tokenize_segment(text, context)
    st.write("Segment tokens shape:", mx.shape(segment_tokens))
    audio_tokens, audio_masks = tokenize_audio(segment_tokens, segment_tokens_mask)
    st.write("Audio tokens shape:", mx.shape(audio_tokens))
    if audio_tokens.ndim < 4:
        audio_tokens = mx.expand_dims(audio_tokens, axis=0)
        if audio_tokens.ndim < 4:
            audio_tokens = mx.expand_dims(audio_tokens, axis=2)
            audio_tokens = mx.expand_dims(audio_tokens, axis=3)
    st.write("Adjusted audio tokens shape:", mx.shape(audio_tokens))
    xs = csm.encoder.encode(audio_tokens)
    st.write("Encoded xs shape:", mx.shape(xs))
    if xs.ndim < 4:
        xs = mx.expand_dims(xs, axis=0)
    pad_width = [(0, 0)] * xs.ndim
    if xs.ndim == 4:
        pad_width[1] = (10, 10)  # Adjust based on debug output; fine-tuned model may need different padding
    xs = mx.pad(xs, pad_width=pad_width, mode="constant")
    st.write("Padded xs shape:", mx.shape(xs))
    try:
        audio = csm.generate(xs, max_length=max_audio_length_ms // 1000, sampler=sampler)
        return audio
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise

def tokenize_segment(text, context):
    import tokenize
    from io import StringIO
    tokens = list(tokenize.generate_tokens(StringIO(text).readline))
    return mx.array(tokens), mx.array([True] * len(tokens))

def tokenize_audio(segment_tokens, segment_masks):
    return segment_tokens, segment_masks