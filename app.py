import streamlit as st
from csm_mlx import CSM, csm_1b, generate, Segment
from mlx_lm.sample_utils import make_sampler
from mlx import nn
import audiofile
import numpy as np
import os
import requests
import mlx.core as mx
import logging
from csm_mlx.generation import generate_frame, make_prompt_cache, decode_audio
from csm_mlx.tokenizers import get_audio_tokenizer

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize and quantize Sesame CSM-1B model with MLX
@st.cache_resource
def load_model():
    try:
        csm = CSM(csm_1b())
        if hasattr(mx, 'is_metal_available') and mx.is_metal_available():
            logging.info("M2 Max GPU (MPS) detected. Using default device for generation.")
        else:
            logging.warning("MPS not available or not supported in this mlx version. Falling back to CPU.")
        
        csm.load_weights("weights/ckpt.safetensors")
        logging.info("Loaded base model")
        return csm
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        st.error(f"Failed to initialize model: {e}")
        return None

csm = load_model()

# Local storage for voice profiles
if 'voice_profiles' not in st.session_state:
    st.session_state.voice_profiles = {}

st.title("Sesame Web App")

# Text input and audio generation
text = st.text_area("Enter text to convert to speech", "")
voice_options = list(st.session_state.voice_profiles.keys()) + ["default"]
voice_id = st.selectbox("Select Voice", voice_options, index=voice_options.index("default") if "default" in voice_options else 0)

if st.button("Generate Audio"):
    sampler = make_sampler(temp=0.8, top_k=50)
    try:
        audio = generate(
            csm, text, speaker=0, context=st.session_state.voice_profiles.get(voice_id, []), max_audio_length_ms=10000, sampler=sampler
        )
        output_path = "output.wav"
        audiofile.write(output_path, np.array(audio), 24000)
        st.session_state['audio_path'] = output_path
        st.audio(output_path, autoplay=True)
        st.download_button("Download Audio", output_path, file_name="output.wav")
    except Exception as e:
        st.error(f"Generation error: {e}")
        logging.error(f"Generation error: {e}")

# Voice cloning
uploaded_file = st.file_uploader("Clone Voice (upload audio)", type=["wav"])
if uploaded_file:
    # Get the filename without extension as the voice ID
    voice_id = os.path.splitext(uploaded_file.name)[0]
    
    temp_path = os.path.join(os.getcwd(), "temp.wav")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    if st.button("Clone Voice"):
        try:
            # Read audio file
            audio_data, sr = audiofile.read(temp_path)
            audio_array = mx.array(audio_data)
            
            # Create voice profile
            voice_profile = Segment(speaker=0, text="Cloned voice", audio=audio_array)
            st.session_state.voice_profiles[voice_id] = [voice_profile]
            st.success(f"Cloned voice with ID: {voice_id}")
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
            logging.error(f"Error processing audio file: {e}")

# Conversation mode
if 'conversation_mode' not in st.session_state:
    st.session_state.conversation_mode = False

if st.button("Start/Stop Conversation"):
    st.session_state.conversation_mode = not st.session_state.conversation_mode

if st.session_state.conversation_mode:
    user_input = st.text_input("Conversation Input")
    if st.button("Send"):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "prompt": "You are an AI, named Gemma, designed to generate plain text responses suitable for text-to-speech conversion. Provide responses that are concise sentences without special formatting. Only include the text to be spoken.\n\n" + user_input,
                    "model": "gemma3:4b",
                    "stream": False
                },
                timeout=10
            )
            response.raise_for_status()
            ollama_response = response.json()
            
            if "response" in ollama_response:
                response_text = ollama_response["response"]
                sampler = make_sampler(temp=0.8, top_k=50)
                audio = generate(
                    csm, response_text, speaker=0, context=st.session_state.voice_profiles.get(voice_id, []), max_audio_length_ms=10000, sampler=sampler
                )
                output_path = "conversation.wav"
                audiofile.write(output_path, np.array(audio), 24000)
                st.session_state['audio_path'] = output_path
                st.audio(output_path, autoplay=True)
                st.download_button("Download Conversation", output_path, file_name="conversation.wav")
            else:
                st.error("Unexpected response format.")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to Ollama API: {e}")
        except Exception as e:
            st.error(f"Conversation error: {e}")
            logging.error(f"General error: {e}")

def generate(csm, text, speaker=0, context=[], max_audio_length_ms=10000, sampler=None):
    try:
        # Tokenize text and context
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = tokenize_segment(
                segment, n_audio_codebooks=csm.n_audio_codebooks
            )
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        # Tokenize the input text
        text_segment_tokens, text_segment_tokens_mask = tokenize_text_segment(text, speaker)
        tokens.append(text_segment_tokens)
        tokens_mask.append(text_segment_tokens_mask)

        # Combine all tokens
        prompt_tokens = mx.concat(tokens, axis=0).astype(mx.int32)
        prompt_tokens_mask = mx.concat(tokens_mask, axis=0)

        # Prepare input for generation
        input = mx.expand_dims(prompt_tokens, 0)
        mask = mx.expand_dims(prompt_tokens_mask, 0)

        # Calculate max sequence length
        max_audio_frames = int(max_audio_length_ms / 80)
        max_seq_len = 2048 - max_audio_frames
        if input.shape[1] >= max_seq_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_audio_frames: {max_seq_len}"
            )

        # Generate audio frames
        samples = []
        backbone_cache = make_prompt_cache(csm.backbone)
        c0_history = []

        for _ in range(max_audio_frames):
            sample = generate_frame(
                csm,
                input,
                sampler=sampler,
                token_mask=mask,
                cache=backbone_cache,
                c0_history=c0_history,
            )

            if not sample.any():
                break  # eos

            samples.append(sample)

            input = mx.expand_dims(mx.concat([sample, mx.zeros((1, 1))], axis=1), 1).astype(
                mx.int32
            )
            mask = mx.expand_dims(
                mx.concat([mx.ones_like(sample), mx.zeros((1, 1))], axis=1), 1
            ).astype(mx.bool_)

        # Decode audio
        audio = (
            decode_audio(
                mx.stack(samples).transpose(1, 2, 0),
                n_audio_codebooks=csm.n_audio_codebooks,
            )
            .squeeze(0)
            .squeeze(0)
        )

        return audio

    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise

def tokenize_segment(segment, n_audio_codebooks=32):
    """Tokenize a segment for the CSM model."""
    if isinstance(segment, Segment):
        # Handle Segment object
        text = segment.text
        audio = segment.audio
        speaker = segment.speaker
    else:
        # Handle raw text
        text = segment
        audio = None
        speaker = 0

    # Tokenize text
    text_tokens = text.split()[:50]  # Limit text length
    text_tokens = mx.array(text_tokens)

    # Create audio tokens if available
    if audio is not None:
        audio_tokens = mx.zeros((len(text_tokens), n_audio_codebooks), dtype=mx.int32)
    else:
        audio_tokens = mx.zeros((len(text_tokens), n_audio_codebooks), dtype=mx.int32)

    # Combine tokens
    tokens = mx.concat([audio_tokens, mx.expand_dims(text_tokens, -1)], axis=-1)
    tokens_mask = mx.ones_like(tokens, dtype=mx.bool_)

    return tokens, tokens_mask

def tokenize_text_segment(text, speaker):
    """Tokenize a text segment for the CSM model."""
    # Simple tokenization - split on whitespace and limit length
    tokens = text.split()[:50]  # Limit text length
    tokens = mx.array(tokens)
    
    # Create audio tokens (zeros for text-only input)
    audio_tokens = mx.zeros((len(tokens), 32), dtype=mx.int32)
    
    # Combine tokens
    tokens = mx.concat([audio_tokens, mx.expand_dims(tokens, -1)], axis=-1)
    tokens_mask = mx.ones_like(tokens, dtype=mx.bool_)
    
    return tokens, tokens_mask