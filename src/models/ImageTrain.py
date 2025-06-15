#!/usr/bin/env python3
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.config.config import config
from io import BytesIO
from torchvision.utils import save_image

# Only configure logging for GAN training progress, loss, and checkpointing
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app_config = config[os.getenv('FLASK_ENV', 'default')]
CHECKPOINT_PATH = app_config.CHECKPOINT_PATH
POKEMON_DATA_PATH = app_config.POKEMON_DATA_PATH
IMAGE_FOLDER_PATH = "data/images"

# Merge data
try:
    metadata = pd.read_csv(POKEMON_DATA_PATH)
    merged_data = []
    for _, row in metadata.iterrows():
        name = row['Name']
        folder_path = os.path.join(IMAGE_FOLDER_PATH, name)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, file)
                    merged_row = row.to_dict()
                    merged_row['image_path'] = image_path
                    merged_data.append(merged_row)
    merged_df = pd.DataFrame(merged_data)
    print(f"[INFO] Successfully merged data. Found {len(merged_df)} images.")
except Exception as e:
    print(f"[ERROR] Error merging data: {e}")
    exit()

#CREATE DATASET FIR FOR PYTORCH
class PokemonDataset(Dataset):
    def __init__(self, df, image_size=256):  # Changed default to 256
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.type1_col = 'Type 1' if 'Type 1' in df.columns else 'Type1'
        self.type2_col = 'Type 2' if 'Type 2' in df.columns else 'Type2'
        self.height_col = 'Height' if 'Height' in df.columns else 'Height (m)'
        self.weight_col = 'Weight' if 'Weight' in df.columns else 'Weight (kg)'
        self.gen_col = 'Generation' if 'Generation' in df.columns else 'Generation'
        self.legendary_col = 'Legendary' if 'Legendary' in df.columns else 'Legendary Status'
        
        for col in [self.type1_col, self.type2_col]:
            if col not in df.columns:
                print(f"Warning: Column {col} not found in dataset. Available columns: {df.columns.tolist()}")
        
        # Get all unique types from both type columns
        self.types = sorted(list(set(
            self.df[self.type1_col].dropna().tolist() + 
            self.df[self.type2_col].dropna().tolist()
        )))
        
        # Get max values for normalization
        self.gen_max = self.df[self.gen_col].max() if self.gen_col in df.columns else 8
        self.height_max = self.df[self.height_col].max() if self.height_col in df.columns else 20.0
        self.weight_max = self.df[self.weight_col].max() if self.weight_col in df.columns else 1000.0

    def one_hot_type(self, t):
        vector = [0] * len(self.types)
        if pd.notna(t) and t in self.types:
            vector[self.types.index(t)] = 1
        return vector
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Create condition vector with proper handling of missing values
        type1 = row.get(self.type1_col, None)
        type2 = row.get(self.type2_col, None)
        height = row.get(self.height_col, 0.0)
        weight = row.get(self.weight_col, 0.0)
        generation = row.get(self.gen_col, 1)
        legendary = row.get(self.legendary_col, False)
        
        # Fix any non-numeric values
        try:
            height = float(height) if pd.notna(height) else 0.0
            weight = float(weight) if pd.notna(weight) else 0.0
            generation = int(generation) if pd.notna(generation) else 1
        except (ValueError, TypeError):
            height, weight, generation = 0.0, 0.0, 1
        
        # Create condition vector
        condition = self.one_hot_type(type1) + self.one_hot_type(type2) + [
            height / self.height_max if self.height_max > 0 else 0,
            weight / self.weight_max if self.weight_max > 0 else 0,
            generation / self.gen_max if self.gen_max > 0 else 0,
            1.0 if legendary else 0.0
        ]
        
        # Load and transform image
        try:
            img = Image.open(row['image_path']).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            logging.warning(f"Error loading image {row['image_path']}: {e}. Returning a black image.")
            img = torch.zeros(3, 256, 256)
            
        return img, torch.tensor(condition).float()

    def __len__(self):
        return len(self.df)

#CREATE GENERATOR
class Generator(nn.Module):
    def __init__(self, z_dim, condition_dim, img_channels=3, feature_g=64):
        super().__init__()
        # Use condition_fc instead of condition_embedding to match checkpoint
        self.condition_fc = nn.Linear(condition_dim, 40)
        
        # The fc layer expects z_dim (100) + processed condition (40) = 140 inputs
        self.fc = nn.Linear(z_dim + 40, feature_g * 16 * 4 * 4)  # 16384 = 64*16*4*4
        
        # Generator backbone with 6 layers for 256x256 output
        self.gen = nn.Sequential(
            nn.BatchNorm2d(feature_g * 16),  # Start with 1024 channels
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 16, feature_g * 8, 4, 2, 1),  # 1024 -> 512 channels, 4x4 -> 8x8
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1),   # 512 -> 256 channels, 8x8 -> 16x16
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1),   # 256 -> 128 channels, 16x16 -> 32x32
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1),       # 128 -> 64 channels, 32x32 -> 64x64
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, feature_g // 2, 4, 2, 1),      # 64 -> 32 channels, 64x64 -> 128x128
            nn.BatchNorm2d(feature_g // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g // 2, img_channels, 4, 2, 1),   # 32 -> 3 channels, 128x128 -> 256x256
            nn.Tanh()
        )

    def forward(self, z, condition):
        # Process condition through the condition_fc layer
        condition_processed = self.condition_fc(condition)
        
        # Concatenate noise and processed condition
        x = torch.cat([z, condition_processed], dim=1)
        
        # Forward through fully connected layer and reshape
        x = self.fc(x).view(-1, 1024, 4, 4)  # Changed to 1024 channels (feature_g * 16)
        
        # Generate image through convolutional layers
        return self.gen(x)
    
##CREATE DISCRIMINATOR TO COMPETE AGAINST GENERATOR
class Discriminator(nn.Module):
    def __init__(self, condition_dim, img_channels=3, feature_d=64):
        super().__init__()
        # Process condition using a fully connected layer
        self.condition_fc = nn.Sequential(
            nn.Linear(condition_dim, 256 * 256),  # Increased for 256x256 images
            nn.LeakyReLU(0.2)
        )
        
        # Main discriminator architecture for 256x256 images
        self.disc = nn.Sequential(
            # Input: 3 channels image + 1 channel condition map = 4 channels
            nn.Conv2d(img_channels + 1, feature_d // 2, 4, 2, 1),       # 4 -> 32 channels, 256x256 -> 128x128
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_d // 2, feature_d, 4, 2, 1),              # 32 -> 64 channels, 128x128 -> 64x64
            nn.BatchNorm2d(feature_d),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1),               # 64 -> 128 channels, 64x64 -> 32x32
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1),           # 128 -> 256 channels, 32x32 -> 16x16
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1),           # 256 -> 512 channels, 16x16 -> 8x8
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_d * 8, feature_d * 16, 4, 2, 1),          # 512 -> 1024 channels, 8x8 -> 4x4
            nn.BatchNorm2d(feature_d * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer to produce 1 value per batch element
            nn.Conv2d(feature_d * 16, 1, 4, 1, 0),                      # 1024 -> 1 channel, 4x4 -> 1x1
            nn.Sigmoid()
        )

    def forward(self, img, condition):
        # Process condition into a 2D feature map (256x256 spatial dimensions)
        condition_map = self.condition_fc(condition).view(-1, 1, 256, 256)
        
        # Concatenate image with condition map along channel dimension
        x = torch.cat([img, condition_map], dim=1)
        
        # Forward through discriminator and flatten output
        return self.disc(x).view(-1, 1)

#TRAIN GENERATIVE ADVERSARIAL NWETWROK
def train_gan(generator, discriminator, dataloader, z_dim, num_epochs=500, start_epoch=0, g_opt=None, d_opt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    criterion = nn.BCELoss()
    
    if g_opt is None:
        g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    if d_opt is None:
        d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=num_epochs)
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=num_epochs)
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        d_losses, g_losses = [], []
        
        if not dataloader:
            logging.warning("Dataloader is empty. Skipping training for this epoch.")
            continue
        
        last_conds = None
        for imgs, conds in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}"):
            last_conds = conds
            imgs, conds = imgs.to(device), conds.to(device)
            bs = imgs.size(0)

            # Train Discriminator
            d_opt.zero_grad()
            z = torch.randn(bs, z_dim).to(device)
            fake_imgs = generator(z, conds)
            d_real = discriminator(imgs, conds)
            d_fake = discriminator(fake_imgs.detach(), conds)
            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            d_loss.backward()
            d_opt.step()
            d_losses.append(d_loss.item())

            # Train Generator
            g_opt.zero_grad()
            z = torch.randn(bs, z_dim).to(device)
            fake_imgs = generator(z, conds)
            d_fake = discriminator(fake_imgs, conds)
            g_loss = criterion(d_fake, torch.ones_like(d_fake))
            g_loss.backward()
            g_opt.step()
            g_losses.append(g_loss.item())

        g_scheduler.step()
        d_scheduler.step()
        
        logging.info(f"Epoch [{epoch+1}] | D Loss: {np.mean(d_losses):.4f} | G Loss: {np.mean(g_losses):.4f}")
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_opt.state_dict(),
            'd_optimizer_state_dict': d_opt.state_dict()
        }, CHECKPOINT_PATH)
        logging.info(f"Saved checkpoint at epoch {epoch+1}")

        if (epoch + 1) % 10 == 0:
            if last_conds is not None:
                with torch.no_grad():
                    sample_noise = torch.randn(1, z_dim).to(device)
                    sample_cond = last_conds[0:1].to(device)
                    sample = generator(sample_noise, sample_cond)
                    buffer = BytesIO()
                    save_image(sample, buffer, format='PNG', normalize=True)
                    buffer.seek(0)
                    logging.info(f"Generated sample image at epoch {epoch+1}")
            else:
                logging.warning("Could not generate sample image because no conditions were available from the dataloader.")

def main():
    dataset = PokemonDataset(merged_df)
    # Pin memory for faster data transfer to GPU
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    _, sample_condition = dataset[0]
    condition_dim = sample_condition.shape[0]
    z_dim = 100

    g = Generator(z_dim=z_dim, condition_dim=condition_dim)
    d = Discriminator(condition_dim=condition_dim)
    
    g_opt = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        try:
            # Use weights_only=False for PyTorch 2.6+ compatibility
            checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
            g.load_state_dict(checkpoint['generator_state_dict'])
            d.load_state_dict(checkpoint['discriminator_state_dict'])
            g_opt.load_state_dict(checkpoint['g_optimizer_state_dict'])
            d_opt.load_state_dict(checkpoint['d_optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Loaded checkpoint. Resuming training from epoch {start_epoch}")
        except (KeyError, TypeError) as e:
            logging.error(f"Checkpoint file is corrupted or has missing keys: {e}. Starting from scratch.")
            # Re-initialize optimizers if checkpoint is bad
            g_opt = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
            d_opt = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    else:
        logging.info("No checkpoint found. Starting training from scratch.")

    train_gan(g, d, dataloader, z_dim, num_epochs=500, start_epoch=start_epoch, g_opt=g_opt, d_opt=d_opt)

if __name__ == "__main__":
    main()