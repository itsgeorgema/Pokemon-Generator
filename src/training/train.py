#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from src.models.ImageTrain import PokemonDataset, Generator, Discriminator
import numpy.core.multiarray
import torch.serialization

# Only configure logging for training progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def train_gan(generator, discriminator, dataloader, z_dim, num_epochs=1000, 
              start_epoch=0, g_opt=None, d_opt=None, checkpoint_path='models/checkpoint.pth', 
              samples_dir=None):
    """
    Train the GAN with improved training techniques for 256x256 images
    """
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    generator.to(device)
    discriminator.to(device)
    
    # Binary cross entropy loss
    criterion = nn.BCELoss()
    
    # Initialize optimizers if not provided
    if g_opt is None:
        g_opt = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    if d_opt is None:
        d_opt = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=num_epochs)
    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_opt, T_max=num_epochs)
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        d_losses, g_losses = [], []
        d_real_losses, d_fake_losses = [], []
        
        if not dataloader:
            logger.warning("Dataloader is empty. Skipping training for this epoch.")
            continue
        
        last_conds = None
        for imgs, conds in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            last_conds = conds
            imgs, conds = imgs.to(device), conds.to(device)
            bs = imgs.size(0)
            
            # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            d_opt.zero_grad()
            
            # Train on real images
            d_real = discriminator(imgs, conds)
            real_labels = torch.ones_like(d_real, device=device)
            # Add some noise to labels for better training stability
            real_labels = real_labels * 0.9  # Label smoothing
            
            d_real_loss = criterion(d_real, real_labels)
            
            # Train on fake images
            z = torch.randn(bs, z_dim, device=device)
            fake_imgs = generator(z, conds)
            d_fake = discriminator(fake_imgs.detach(), conds)
            fake_labels = torch.zeros_like(d_fake, device=device)
            
            d_fake_loss = criterion(d_fake, fake_labels)
            
            # Combine discriminator losses and update
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_opt.step()
            d_losses.append(d_loss.item())
            d_real_losses.append(d_real_loss.item())
            d_fake_losses.append(d_fake_loss.item())
            
            # Train Generator: maximize log(D(G(z)))
            g_opt.zero_grad()
            
            # Generate new fake images (reusing noise from above is also ok)
            z = torch.randn(bs, z_dim, device=device)
            fake_imgs = generator(z, conds)
            d_fake = discriminator(fake_imgs, conds)
            
            g_loss = criterion(d_fake, real_labels)
            g_loss.backward()
            g_opt.step()
            g_losses.append(g_loss.item())
        
        g_scheduler.step()
        d_scheduler.step()
        
        # Calculate average losses for this epoch
        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)
        avg_d_real_loss = np.mean(d_real_losses)
        avg_d_fake_loss = np.mean(d_fake_losses)
        
        # Get current learning rates
        g_lr = g_opt.param_groups[0]['lr']
        d_lr = d_opt.param_groups[0]['lr']
        
        # Log detailed information for every epoch
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"D Loss: {avg_d_loss:.4f} (Real: {avg_d_real_loss:.4f}, Fake: {avg_d_fake_loss:.4f}) | "
            f"G Loss: {avg_g_loss:.4f} | "
            f"LR: G={g_lr:.6f}, D={d_lr:.6f}"
        )
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_opt.state_dict(),
            'd_optimizer_state_dict': d_opt.state_dict(),
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'd_real_loss': avg_d_real_loss,
            'd_fake_loss': avg_d_fake_loss
        }, checkpoint_path)
        logger.info(f"Saved checkpoint at epoch {epoch+1}")
        
        # Generate and save a sample image every 10 epochs
        if (epoch + 1) % 10 == 0 and last_conds is not None:
            with torch.no_grad():
                n = min(8, last_conds.shape[0])
                sample_noise = torch.randn(n, z_dim, device=device)
                sample_conds = last_conds[0:n].to(device)
                samples = generator(sample_noise, sample_conds)
                logger.info(f"Generated sample images at epoch {epoch+1} (not saved to disk)")

def main():
    parser = argparse.ArgumentParser(description='Train Pokemon GAN for 256x256 images')
    parser.add_argument('--data_path', type=str, default='data/Pokemon_stats.csv', help='Path to Pokemon data CSV')
    parser.add_argument('--image_folder', type=str, default='data/images', help='Path to folder containing Pokemon images')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training (reduced for larger images)')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoint.pth', help='Path to save/load checkpoint')
    parser.add_argument('--samples_dir', type=str, default='static/samples', help='Directory to save sample images')
    parser.add_argument('--image_size', type=int, default=256, help='Size of generated images')
    parser.add_argument('--z_dim', type=int, default=100, help='Dimension of noise vector')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Load data
    print(f"[INFO] Loading data from {args.data_path}")
    try:
        metadata = pd.read_csv(args.data_path)
        print(f"[INFO] Loaded {len(metadata)} Pokemon from {args.data_path}")
        print(f"[INFO] Columns: {metadata.columns.tolist()}")
        name_col = None
        if 'Name' in metadata.columns:
            name_col = 'Name'
        elif 'name' in metadata.columns:
            name_col = 'name'
        
        if name_col is None:
            print("[ERROR] No name column found in the dataset!")
            return

        merged_data = []
        for idx, row in metadata.iterrows():
            name = row[name_col]
            if pd.notna(name):
                folder_path = os.path.join(args.image_folder, name)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(folder_path, file)
                            merged_row = row.to_dict()
                            merged_row['image_path'] = image_path
                            merged_data.append(merged_row)
        
        merged_df = pd.DataFrame(merged_data)
        print(f"[INFO] Successfully merged data. Found {len(merged_df)} images.")
        
        # Create dataset and dataloader
        dataset = PokemonDataset(merged_df, image_size=args.image_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        if len(dataset) == 0:
            print("[ERROR] No valid Pokemon images found. Please check your image folder and data paths.")
            return
        
        # Get sample condition vector to determine dimension
        _, sample_condition = dataset[0]
        condition_dim = sample_condition.shape[0]
        print(f"[INFO] Condition dimension: {condition_dim}")
        
        # Initialize models for 256x256 output
        g = Generator(z_dim=args.z_dim, condition_dim=condition_dim)
        d = Discriminator(condition_dim=condition_dim)
        
        # Initialize optimizers
        g_opt = optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
        d_opt = optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
        
        start_epoch = 0
        
        # Load checkpoint if resume flag is set
        if args.resume and os.path.exists(args.checkpoint):
            try:
                print(f"[INFO] Loading checkpoint from {args.checkpoint}")
                try:
                    # First try with weights_only=False (for PyTorch 2.6+ compatibility)
                    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'), weights_only=False)
                except Exception as e1:
                    try:
                        # Second try with pickle_module=torch.serialization.pickle
                        import pickle
                        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'), 
                                               pickle_module=pickle, weights_only=False)
                    except Exception as e2:
                        try:
                            # Third try with allowlisting
                            torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
                            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'), weights_only=False)
                        except Exception as e3:
                            # Fourth try with pickle4 compatibility for "invalid load key, 'v'" error
                            import pickle
                            import io
                            
                            # Custom unpickler class for compatibility with old formats
                            class LegacyUnpickler(pickle.Unpickler):
                                def find_class(self, module, name):
                                    if module == 'collections' and name == 'OrderedDict':
                                        return dict
                                    return super().find_class(module, name)
                            
                            # Load the file manually and use our custom unpickler
                            with open(args.checkpoint, 'rb') as f:
                                checkpoint = LegacyUnpickler(f).load()
                                
                            print("[INFO] Successfully loaded checkpoint with custom legacy unpickler")
                
                g.load_state_dict(checkpoint['generator_state_dict'], strict=False)
                d.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
                g_opt.load_state_dict(checkpoint['g_optimizer_state_dict'])
                d_opt.load_state_dict(checkpoint['d_optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"[INFO] Resuming training from epoch {start_epoch}")
            except (KeyError, TypeError) as e:
                print(f"[ERROR] Checkpoint file is corrupted or has missing keys: {e}. Starting from scratch.")
                g_opt = optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
                d_opt = optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
                start_epoch = 0
        else:
            print("[INFO] Starting training from scratch for 256x256 images")
        
        # Start training
        train_gan(g, d, dataloader, args.z_dim, 
                  num_epochs=args.epochs,
                  start_epoch=start_epoch,
                  g_opt=g_opt, 
                  d_opt=d_opt, 
                  checkpoint_path=args.checkpoint,
                  samples_dir=args.samples_dir)
        
    except Exception as e:
        print(f"[ERROR] Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 