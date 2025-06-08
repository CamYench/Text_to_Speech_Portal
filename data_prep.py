import os
import numpy as np
import audiofile
import mlx.core as mx
from whisper import load_model, transcribe
from pathlib import Path
import urllib.request
import ssl
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_transcript(wav_path):
    """Generate transcript using Whisper ASR model with SSL bypass."""
    try:
        # Create an unverified SSL context
        context = ssl._create_unverified_context()
        # Create a custom opener with the unverified context
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
        urllib.request.install_opener(opener)
        
        # Load model with explicit device and dtype settings
        model = load_model("base", device="cpu", download_root=os.path.expanduser("~/.cache/whisper"))
        logging.info("Successfully loaded Whisper model")
        
        result = transcribe(model, wav_path, fp16=False)  # Explicitly disable FP16
        segments = result["segments"]
        transcript = {i: {"text": seg["text"], "start": seg["start"], "end": seg["end"]} for i, seg in enumerate(segments)}
        logging.info(f"Successfully generated transcript with {len(segments)} segments")
        return transcript
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise

def format_for_finetuning(wav_path, output_dir="segments", max_length_ms=10000):
    """Segment WAV file and format for CSMDataset."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Processing audio file: {wav_path}")
        
        transcript = generate_transcript(wav_path)
        audio_data, sr = audiofile.read(wav_path)
        
        # Ensure audio data is in the correct format
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono if stereo
        
        # Convert to float32 if not already
        audio_data = audio_data.astype(np.float32)
        
        # Normalize audio data to prevent clipping
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        audio_array = mx.array(audio_data)
        duration_ms = len(audio_data) / sr * 1000
        
        # Create a single conversation with all segments
        conversation = []
        
        for idx, (i, seg) in enumerate(transcript.items()):
            start_ms = int(seg["start"] * 1000)
            end_ms = min(int(seg["end"] * 1000), start_ms + max_length_ms)
            if end_ms > start_ms:
                # Calculate sample indices
                start_sample = int(start_ms / 1000 * sr)
                end_sample = int(end_ms / 1000 * sr)
                
                # Extract segment and convert to numpy array
                segment_data = audio_array[start_sample:end_sample]
                segment_data_np = np.array(segment_data)
                
                # Ensure segment data is in the correct format
                segment_data_np = segment_data_np.astype(np.float32)
                
                segment_path = os.path.join(output_dir, f"turn{idx}.wav")
                try:
                    audiofile.write(segment_path, segment_data_np, sr)
                    logging.info(f"Created segment {idx} from {start_ms}ms to {end_ms}ms")
                    conversation.append({
                        "text": seg["text"],
                        "audio_path": segment_path,
                        "speaker": 0  # Using single speaker for voice cloning
                    })
                except Exception as e:
                    logging.error(f"Error writing segment {idx}: {e}")
                    continue

        if not conversation:
            raise ValueError("No valid segments were created")

        # Format as a list of conversations (each conversation is a list of segments)
        dataset = [conversation]  # Single conversation for voice cloning
        
        dataset_path = os.path.join(output_dir, "dataset.json")
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)
        logging.info(f"Dataset prepared successfully at: {dataset_path}")
        return dataset_path
    except Exception as e:
        logging.error(f"Error during data preparation: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python data_prep.py <path_to_wav_file>")
        sys.exit(1)
    wav_path = sys.argv[1]
    dataset_path = format_for_finetuning(wav_path)
    print(f"Dataset prepared at: {dataset_path}")