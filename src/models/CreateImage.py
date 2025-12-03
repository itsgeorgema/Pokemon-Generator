#!/usr/bin/env python3
import torch
import os
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision.utils import save_image
from io import BytesIO
import torch.serialization
import numpy.core.multiarray

def _load_metadata_defaults():
    """
    Lazily load metadata used for normalization.
    This avoids doing pandas I/O and preprocessing at module import time,
    which helps reduce application startup latency.
    """
    try:
        metadata = pd.read_csv("data/Pokemon_stats.csv")
        type1_col = 'Type 1' if 'Type 1' in metadata.columns else 'Type1'
        type2_col = 'Type 2' if 'Type 2' in metadata.columns else 'Type2'
        height_col = 'Height' if 'Height' in metadata.columns else 'Height (m)'
        weight_col = 'Weight' if 'Weight' in metadata.columns else 'Weight (kg)'
        gen_col = 'Generation' if 'Generation' in metadata.columns else 'Generation'

        types = sorted(list(set(metadata[type1_col].dropna().tolist() + metadata[type2_col].dropna().tolist())))
        max_height = metadata[height_col].max() if height_col in metadata.columns else 20.0
        max_weight = metadata[weight_col].max() if weight_col in metadata.columns else 1000.0
        max_gen = metadata[gen_col].max() if gen_col in metadata.columns else 8
    except Exception as e:
        print(f"Warning: Error loading metadata: {e}. Using default values.")
        types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting',
                 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost',
                 'Dragon', 'Dark', 'Steel', 'Fairy']
        max_height = 20.0
        max_weight = 1000.0
        max_gen = 8

    return types, max_height, max_weight, max_gen

def one_hot_type(t, types):
    vector = [0] * len(types)
    if pd.notna(t) and t in types:
        vector[types.index(t)] = 1
    return vector

class Generator(nn.Module):
    def __init__(self, z_dim, condition_dim, img_channels=3, feature_g=64):
        super().__init__()
        self.condition_fc = nn.Linear(condition_dim, 40)
        self.fc = nn.Linear(z_dim + 40, feature_g * 16 * 4 * 4)  # Increased initial features
        
        self.gen = nn.Sequential(
            nn.BatchNorm2d(feature_g * 16),  # Start with 1024 channels
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 16, feature_g * 8, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1),   # 8x8 -> 16x16
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1),   # 16x16 -> 32x32
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1),       # 32x32 -> 64x64
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, feature_g // 2, 4, 2, 1),      # 64x64 -> 128x128
            nn.BatchNorm2d(feature_g // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g // 2, img_channels, 4, 2, 1),   # 128x128 -> 256x256
            nn.Tanh()
        )

    def forward(self, z, condition):
        condition_processed = self.condition_fc(condition)
        x = torch.cat([z, condition_processed], dim=1)
        x = self.fc(x).view(-1, 1024, 4, 4)  # Changed to 1024 channels (feature_g * 16)
        return self.gen(x)

def generate_image(generator, condition, z_dim=100):
    """Generate a single Pokemon image with the given condition"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    condition = condition.to(device)
    
    z = torch.randn(1, z_dim, device=device)
    
    with torch.no_grad():
        fake_img = generator(z, condition)
    
    return fake_img[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5

def create_fallback_image(type1, type2=None):
    """Create a colored image based on Pokemon type when the model fails"""
    image = np.ones((256, 256, 3), dtype=np.float32)
    
    type_colors = {
        'normal': [0.7, 0.7, 0.5],
        'fire': [0.9, 0.4, 0.1],
        'water': [0.4, 0.5, 0.9],
        'electric': [0.9, 0.8, 0.1],
        'grass': [0.5, 0.8, 0.3],
        'ice': [0.6, 0.8, 0.8],
        'fighting': [0.8, 0.2, 0.2],
        'poison': [0.7, 0.3, 0.7],
        'ground': [0.8, 0.7, 0.3],
        'flying': [0.6, 0.6, 0.9],
        'psychic': [0.9, 0.4, 0.5],
        'bug': [0.7, 0.8, 0.1],
        'rock': [0.7, 0.6, 0.3],
        'ghost': [0.5, 0.4, 0.7],
        'dragon': [0.5, 0.3, 0.9],
        'dark': [0.5, 0.4, 0.3],
        'steel': [0.7, 0.7, 0.8],
        'fairy': [0.9, 0.6, 0.7]
    }
    
    color1 = type_colors.get(type1.lower(), [0.7, 0.7, 0.7])
    
    image[:, :, 0] = color1[0]
    image[:, :, 1] = color1[1]
    image[:, :, 2] = color1[2]
    
    if type2:
        color2 = type_colors.get(type2.lower(), [0.7, 0.7, 0.7])
        
        for i in range(256):
            for j in range(256):
                if (i + j) % 16 < 8:  # Adjusted pattern scale for 256x256
                    image[i, j, 0] = color2[0]
                    image[i, j, 1] = color2[1]
                    image[i, j, 2] = color2[2]
    
    image = torch.from_numpy(image.transpose(2, 0, 1))
    image = image.unsqueeze(0)
    
    return image

def generate_and_save_image(type1, type2, height, weight, generation, legendary,
                            checkpoint_path="models/checkpoint.pth",
                            data_path="data/Pokemon_stats.csv"):
    """Generate a Pokemon image using the GAN model and return image bytes."""
    try:
        metadata = pd.read_csv(data_path)
        type1_col = 'Type 1' if 'Type 1' in metadata.columns else 'Type1'
        type2_col = 'Type 2' if 'Type 2' in metadata.columns else 'Type2'
        types = sorted(list(set(metadata[type1_col].dropna().tolist() + metadata[type2_col].dropna().tolist())))
        
        # More robust column name handling
        height_col = next((col for col in ['Height', 'Height (m)'] if col in metadata.columns), None)
        weight_col = next((col for col in ['Weight', 'Weight (kg)'] if col in metadata.columns), None)
        
        max_height = metadata[height_col].max() if height_col else 20.0
        max_weight = metadata[weight_col].max() if weight_col else 1000.0
        max_gen = metadata['Generation'].max() if 'Generation' in metadata.columns else 8
    except Exception as e:
        print(f"Warning: Using default type list: {e}")
        types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', \
                'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', \
                'Dragon', 'Dark', 'Steel', 'Fairy']
        max_height = 20.0
        max_weight = 1000.0
        max_gen = 8
    try:
        # Prepare condition vector
        type1_vec = one_hot_type(type1, types)
        type2_vec = one_hot_type(type2, types) if type2 else [0] * len(types)
        condition_vec = type1_vec + type2_vec + [
            min(height, max_height) / max_height,
            min(weight, max_weight) / max_weight,
            min(generation, max_gen) / max_gen,
            1.0 if legendary else 0.0
        ]
        condition = torch.tensor([condition_vec], dtype=torch.float32)
        z_dim = 100
        condition_dim = len(condition_vec)
        generator = Generator(z_dim=z_dim, condition_dim=condition_dim)
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found at {checkpoint_path}. Using fallback image.")
            fallback_img = create_fallback_image(type1, type2)
            buffer = BytesIO()
            save_image(fallback_img, buffer, format='PNG', normalize=True)
            buffer.seek(0)
            return buffer.read()
            
        # Load checkpoint and generate image
        try:
            # Fix for PyTorch version incompatibility issues
            try:
                # First try with weights_only=False (for PyTorch 2.6+ compatibility)
                checkpoint = torch.load(
                    checkpoint_path, 
                    map_location=torch.device('cpu'),
                    weights_only=False
                )
            except Exception as e1:
                try:
                    # Second try with pickle_module=torch.serialization.pickle
                    import pickle
                    checkpoint = torch.load(
                        checkpoint_path, 
                        map_location=torch.device('cpu'),
                        pickle_module=pickle,
                        weights_only=False
                    )
                except Exception as e2:
                    try:
                        # Third try with allowlisting
                        torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
                        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
                    except Exception as e3:
                        print(f"Normal loading methods failed: {e3}. Trying direct method...")
                        # Final attempt - use a direct approach with fixed parameters for CI/CD
                        try:
                            # Use Python's builtin pickle with specific options for old checkpoints
                            import pickle
                            import io

                            with open(checkpoint_path, 'rb') as f:
                                data = f.read()
                                
                                # Try to load directly with modified pickle parameters
                                buffer = io.BytesIO(data)
                                checkpoint = torch.load(
                                    buffer, 
                                    map_location='cpu',
                                    pickle_module=pickle,
                                    weights_only=False,
                                    _pickle_load_args={'fix_imports': True, 'encoding': 'latin1'}
                                )
                        except Exception as e4:
                            print(f"Checkpoint loading failed after multiple attempts: {e4}")
                            raise e4
                            
            if "generator_state_dict" not in checkpoint:
                print("Invalid checkpoint format. Using fallback image.")
                fallback_img = create_fallback_image(type1, type2)
                buffer = BytesIO()
                save_image(fallback_img, buffer, format='PNG', normalize=True)
                buffer.seek(0)
                print("[GAN Fallback] Used fallback image due to invalid checkpoint format.")
                return buffer.read()
            
            # Always try to load state dict without special CI/CD handling
            try:
                generator.load_state_dict(checkpoint["generator_state_dict"], strict=False)
                print("Successfully loaded model weights from checkpoint")
            except Exception as e:
                print(f"Error during state dict loading: {e}")
                # If loading fails but we need to use the model anyway,
                # use whatever parameters were already loaded
                    
            generator.eval()
            with torch.no_grad():
                z = torch.randn(1, z_dim)
                fake_img = generator(z, condition)
            buffer = BytesIO()
            save_image(fake_img, buffer, format='PNG', normalize=True)
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            print(f"Error generating image with GAN: {e}. Using fallback.")
            fallback_img = create_fallback_image(type1, type2)
            buffer = BytesIO()
            save_image(fallback_img, buffer, format='PNG', normalize=True)
            buffer.seek(0)
            print("[GAN Fallback] Used fallback image due to GAN error.")
            return buffer.read()
    except Exception as e:
        print(f"Unexpected error in generate_and_save_image: {e}")
        fallback_img = create_fallback_image(type1, type2)
        buffer = BytesIO()
        save_image(fallback_img, buffer, format='PNG', normalize=True)
        buffer.seek(0)
        return buffer.read()
