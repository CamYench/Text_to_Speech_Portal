import os
import shutil
from csm_mlx import CSM, csm_1b, load_adapters, Segment
from csm_mlx.finetune import CSMTrainer, TrainArgs, CSMDataset
from csm_mlx.finetune.utils import linear_to_lora_layers
from mlx.optimizers import AdamW
from pathlib import Path
import json
import logging
import mlx.core as mx
import audiofile
from csm_mlx.tokenizers import get_audio_tokenizer
from moshi_mlx.models.mimi import Mimi, mimi_202407
from mlx_lm.utils import save_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_disk_space(path, required_space_mb=100):
    """Check if there's enough disk space available."""
    try:
        total, used, free = shutil.disk_usage(path)
        free_mb = free // (1024 * 1024)  # Convert to MB
        if free_mb < required_space_mb:
            raise RuntimeError(f"Not enough disk space. Required: {required_space_mb}MB, Available: {free_mb}MB")
        logging.info(f"Available disk space: {free_mb}MB")
        return True
    except Exception as e:
        logging.error(f"Error checking disk space: {e}")
        raise

def find_dataset_file(dataset_path):
    """Find the dataset file in either segments or weights directory."""
    # Check if the path is absolute or relative
    if os.path.isabs(dataset_path):
        if os.path.exists(dataset_path):
            return dataset_path
    
    # Check in segments directory
    segments_path = os.path.join("segments", "dataset.json")
    if os.path.exists(segments_path):
        return segments_path
    
    # Check in weights directory
    weights_path = os.path.join("weights", "dataset.json")
    if os.path.exists(weights_path):
        return weights_path
    
    raise FileNotFoundError(f"Dataset not found in {dataset_path}, segments/dataset.json, or weights/dataset.json")

def finetune_and_save_profile(dataset_path, output_dir="finetuned_weights", voice_id="custom"):
    """Fine-tune CSM using available hardware and save as a persistent voice profile."""
    logging.info(f"Starting fine-tuning for voice profile {voice_id}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.abspath(output_dir)  # Convert to absolute path
    os.makedirs(output_dir, exist_ok=True)
    
    # Check disk space before starting
    check_disk_space(output_dir)
    
    # Check if MPS is available and set device
    if hasattr(mx, 'is_metal_available') and mx.is_metal_available():
        logging.info("M2 Max GPU (MPS) detected. Using default device.")
    else:
        logging.warning("MPS not available or not supported in this mlx version. Using CPU.")
    
    try:
        # Load model and config
        logging.info("Loading base model and config...")
        csm = CSM(csm_1b())
        csm.load_weights("weights/ckpt.safetensors")
        
        # Load model config
        config_path = os.path.join("weights", "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found at {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        logging.info("Successfully loaded model config")
        
        # Freeze base model parameters
        logging.info("Freezing base model parameters...")
        for param in csm.parameters():
            if isinstance(param, mx.array):
                param.requires_grad = False
            
        # Convert to LoRA for memory efficiency
        logging.info("Converting to LoRA layers...")
        # Convert linear layers to LoRA with model config
        linear_to_lora_layers(csm, config)
        
        # Enable gradients for LoRA parameters
        for param in csm.parameters():
            if isinstance(param, mx.array) and "lora" in str(param):
                param.requires_grad = True
                logging.info(f"Enabled gradients for LoRA parameter")

        # Find and load dataset
        logging.info(f"Looking for dataset file...")
        dataset_file = find_dataset_file(dataset_path)
        logging.info(f"Loading dataset from {dataset_file}...")
        
        with open(dataset_file, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Dataset must be a list of conversations")
        logging.info(f"Loaded {len(data)} conversations")
        
        # Convert data to list of lists of Segment objects
        conversations = []
        for conversation in data:
            if not isinstance(conversation, list):
                raise ValueError(f"Invalid conversation format: {conversation}")
            
            conversation_segments = []
            for item in conversation:
                if not isinstance(item, dict) or 'text' not in item or 'audio_path' not in item:
                    raise ValueError(f"Invalid segment format: {item}")
                
                # Load audio data
                audio_data, sr = audiofile.read(item['audio_path'])
                audio_array = mx.array(audio_data)
                
                # Create Segment object
                segment = Segment(
                    speaker=item.get('speaker', 0),
                    text=item['text'],
                    audio=audio_array
                )
                conversation_segments.append(segment)
            conversations.append(conversation_segments)
        
        # Create dataset with proper format
        dataset = CSMDataset(conversations, n_audio_codebooks=32, max_audio_length_ms=10000)

        # Training arguments optimized for available hardware
        train_args = TrainArgs(
            model=csm,
            optimizer=AdamW(learning_rate=1e-4),
            output_dir=Path(output_dir),  # Use the absolute path
            first_codebook_weight_multiplier=1.0,
            max_norm=1.0,
            gradient_checkpointing=True,
            log_freq=10,
            ckpt_freq=50,
            only_save_trainable_params=True
        )

        # Initialize and train
        logging.info("Initializing trainer...")
        trainer = CSMTrainer(train_args)
        logging.info("Starting training...")
        trainer.train(dataset, batch_size=4, epochs=3)
        logging.info("Training completed successfully.")

        # Save adapters and profile config
        adapters_path = os.path.join(output_dir, "adapters.safetensors")
        config_path = os.path.join(output_dir, "profile_config.json")
        
        logging.info("Saving profile configuration...")
        with open(config_path, "w") as f:
            json.dump({
                "voice_id": voice_id,
                "path": output_dir,
                "training_info": {
                    "num_segments": len(conversations),
                    "device": "mps" if hasattr(mx, 'is_metal_available') and mx.is_metal_available() else "cpu"
                }
            }, f, indent=2)
        
        # Save adapter config for LoRA
        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": config.get("num_layers", 16),
            "lora_parameters": {
                "rank": 8,
                "dropout": 0.0,
                "scale": 10.0
            }
        }
        save_config(adapter_config, os.path.join(output_dir, "adapter_config.json"))
        
        logging.info(f"Profile saved successfully at: {output_dir}")
        return output_dir
        
    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune CSM model for voice cloning")
    parser.add_argument("dataset_path", help="Path to the dataset JSON file")
    parser.add_argument("--output-dir", default="finetuned_weights", help="Directory to save the fine-tuned model")
    parser.add_argument("--voice-id", default="custom", help="ID for the voice profile")
    args = parser.parse_args()
    
    output_dir = finetune_and_save_profile(args.dataset_path, args.output_dir, args.voice_id)
    print(f"Fine-tuning completed. Weights saved at: {output_dir}")