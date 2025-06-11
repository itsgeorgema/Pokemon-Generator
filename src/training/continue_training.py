#!/usr/bin/env python3
"""
Script to continue training the Pokemon Generator model from the last saved checkpoint.
This script will automatically detect the last epoch and continue training from there.
"""
import os
import torch
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Continue training Pokemon GAN from last checkpoint')
    parser.add_argument('--data_path', type=str, default='data/Pokemon_stats.csv', help='Path to Pokemon data CSV')
    parser.add_argument('--image_folder', type=str, default='data/images', help='Path to folder containing Pokemon images')
    parser.add_argument('--epochs', type=int, default=100, help='Number of additional training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoint.pth', help='Path to load checkpoint')
    parser.add_argument('--backup', action='store_true', help='Create a backup of the current checkpoint')
    
    args = parser.parse_args()
    
    # Create a backup of the current checkpoint if requested
    if args.backup and os.path.exists(args.checkpoint):
        import shutil
        backup_path = f"{args.checkpoint}.backup"
        shutil.copy2(args.checkpoint, backup_path)
        logging.info(f"Created backup of current checkpoint at {backup_path}")
    
    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint file {args.checkpoint} not found!")
        return
    
    # Load the checkpoint to get the last epoch
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        last_epoch = checkpoint.get('epoch', 0)
        logging.info(f"Found checkpoint at epoch {last_epoch}")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return
    
    # Calculate total epochs
    total_epochs = last_epoch + 1 + args.epochs
    
    # Build the command to run train.py with resume flag
    cmd = [
        "python", "-m", "src.training.train",
        "--data_path", args.data_path,
        "--image_folder", args.image_folder,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--checkpoint", args.checkpoint,
        "--resume"  # This flag tells train.py to load the checkpoint
    ]
    
    # Execute the command
    import subprocess
    logging.info(f"Starting training from epoch {last_epoch + 1} for {args.epochs} more epochs (will end at epoch {total_epochs})")
    logging.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"Training completed successfully! Reached epoch {total_epochs}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed with exit code {e.returncode}")
    
if __name__ == "__main__":
    main() 