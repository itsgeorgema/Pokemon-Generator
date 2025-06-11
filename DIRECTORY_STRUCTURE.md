# Pokemon Generator Directory Structure

This document outlines the organization of the Pokemon Generator codebase and the migration process.

## Directory Structure

```
Pokemon-Generator/
├── app.py                    # Main Flask application
├── main.py                   # Entry point for the application
├── run_local.py              # Simplified script for local testing without database
├── src/                      # Source code directory
│   ├── __init__.py           # Package initialization
│   ├── config/               # Configuration module
│   │   ├── __init__.py       # Package initialization
│   │   └── config.py         # Application configuration
│   ├── models/               # Model definitions
│   │   ├── __init__.py       # Package initialization
│   │   ├── CreateImage.py    # GAN model for image generation
│   │   └── ImageTrain.py     # Code for training the GAN
│   ├── training/             # Training scripts
│   │   ├── __init__.py       # Package initialization
│   │   ├── train.py          # Main training script for 256x256 images
│   │   └── continue_training.py # Script to continue training from checkpoint
│   └── utils/                # Utility functions
│       ├── __init__.py       # Package initialization
│       ├── check_data.py     # Data validation utilities
│       └── monitoring.py     # Performance monitoring utilities
├── templates/                # HTML templates
├── static/                   # Static files
│   ├── generated/            # Generated Pokemon images
│   └── samples/              # Sample images from training
├── models/                   # Model checkpoints
│   └── checkpoint.pth        # Trained GAN model
├── data/                     # Data files
│   ├── images/               # Pokemon training images
│   ├── Pokemon_stats.csv     # Pokemon statistics data
│   └── pokemon_data_pokeapi.csv # Pokemon API data
└── logs/                     # Log files
├── scripts/                  # Utility scripts for running, deploying, and database migration
│   ├── start_server.sh       # Start the server
│   ├── init_db.py            # Initialize the database
│   ├── deploy_render.sh      # Deploy to Render
│   ├── deploy_to_render.sh   # Alternate deploy script
│   ├── db_migrate.py         # Database migration utility
│   └── README.md             # Scripts documentation
```

## Migration Process

The code migration involved the following steps:

1. **Create the proper directory structure**:
   - Created `src/` directory with subdirectories for modules
   - Created `__init__.py` files to make them proper Python packages

2. **Moved files to appropriate locations**:
   - Moved model files to `src/models/`
   - Moved training scripts to `src/training/`
   - Moved configuration to `src/config/`
   - Moved data files to `data/`
   - Moved checkpoints to `models/`

3. **Updated import paths**:
   - Updated import statements in all files to use the new package structure
   - Fixed paths to data files and checkpoints

4. **Created support scripts**:
   - `main.py`: New entry point for the application
   - `run_local.py`: Simplified version for local testing without database

5. **Updated documentation**:
   - Updated README.md with new command paths and directory structure
   - Created DIRECTORY_STRUCTURE.md to document the organization

## Notes on Compatibility

The codebase was migrated to use the new directory structure while maintaining backward compatibility with existing checkpoints. However, the GAN architecture has been updated to generate 256x256 images, which means older checkpoints trained for 64x64 images will show a model size mismatch. Thus, the old checkpoint file was deleted, and the model was retrained.