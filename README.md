# Pokemon Generator

A web application that generates unique Pokemon images using a conditional GAN (Generative Adversarial Network) based on user-selected types and other parameters.

## Features

- Generate unique Pokemon images based on selected types
- Customize generation with legendary status, height, weight, and generation parameters
- View predicted stats based on Pokemon types
- Responsive web interface
- Docker support for easy deployment
- PostgreSQL database for storing generated images

## Tech Stack

### Backend
- **Python 3.9+**: Core programming language
- **Flask**: Web framework for the API
- **SQLAlchemy**: ORM for database operations
- **PostgreSQL**: Database for storing generated images
- **PyTorch**: Deep learning framework for the GAN model
- **scikit-learn**: For training stat prediction models
- **pandas & numpy**: Data processing and manipulation

### Frontend
- **HTML/CSS/JavaScript**: Frontend interface
- **Tailwind CSS**: For frontend styling

### Deployment & Infrastructure
- **Docker & Docker Compose**: Containerization
- **Gunicorn**: WSGI HTTP Server
- **Render**: Cloud deployment platform

## GAN Architecture

The Pokemon Generator uses a conditional GAN architecture specifically designed to generate 256x256 pixel Pokemon images based on type and other attributes.

### Generator Architecture

The generator takes a random noise vector (z_dim=100) and a condition vector as input and produces a 256x256 RGB image:

1. **Condition Processing**: The condition vector (containing type, height, weight, generation, and legendary status) is processed through a fully connected layer to create a 40-dimensional embedding.

2. **Initial Projection**: The noise vector and processed condition are concatenated and projected to a 4×4×1024 feature map using a fully connected layer.

3. **Upsampling Layers**: Six transposed convolution layers progressively upsample the feature map:
   - 4×4×1024 → 8×8×512
   - 8×8×512 → 16×16×256
   - 16×16×256 → 32×32×128
   - 32×32×128 → 64×64×64
   - 64×64×64 → 128×128×32
   - 128×128×32 → 256×256×3

4. **Activation**: Each layer uses BatchNorm and ReLU activation, with a final Tanh activation to produce pixel values in the range [-1, 1].

### Discriminator Architecture

The discriminator evaluates whether an image is real or generated:

1. **Condition Processing**: The condition vector is processed through a fully connected layer and reshaped to a 256×256×1 feature map.

2. **Image Processing**: The input image (256×256×3) is concatenated with the condition map (256×256×1) to form a 256×256×4 input.

3. **Downsampling Layers**: Six convolutional layers progressively downsample the image:
   - 256×256×4 → 128×128×32
   - 128×128×32 → 64×64×64
   - 64×64×64 → 32×32×128
   - 32×32×128 → 16×16×256
   - 16×16×256 → 8×8×512
   - 8×8×512 → 4×4×1024
   - 4×4×1024 → 1×1×1

4. **Activation**: LeakyReLU activation is used throughout, with BatchNorm for normalization and a final Sigmoid activation to output a probability.

## Training the GAN

The GAN is trained using a standard adversarial process with several optimizations for stability:

### Training Process

1. **Data Preparation**: Pokemon images are loaded and paired with their type, height, weight, generation, and legendary status.

2. **Condition Vector**: For each Pokemon, a condition vector is created by:
   - One-hot encoding the primary and secondary types
   - Normalizing height, weight, and generation
   - Adding a binary flag for legendary status

3. **Training Loop**:
   - The discriminator is trained to classify real and fake images
   - The generator is trained to produce images that fool the discriminator
   - Label smoothing (0.9 for real labels) is used to improve stability
   - Adam optimizer with learning rate 2e-4 and betas (0.5, 0.999) is used
   - Cosine annealing learning rate scheduling is applied

4. **Checkpointing**: Model checkpoints are saved after each epoch, allowing training to be resumed if interrupted.

### Training Command

```bash
# Start training from scratch
python -m src.training.train --epochs 1000 --batch_size 16 --image_size 256

# Resume training from a checkpoint
python -m src.training.continue_training --epochs 100 --checkpoint models/checkpoint.pth
```

## Stat Prediction

The application uses scikit-learn's RandomForestRegressor models to predict Pokemon stats based on their types:

1. For each stat (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed), a separate RandomForestRegressor model is trained on type combinations.
2. Training data is derived from the average stats of existing Pokemon with each type combination.
3. The models predict reasonable stat values for new type combinations, which are then displayed in the UI.

## Environment Setup

The application uses environment variables for configuration. These must be set in a `.env` file in the root directory.

Here are all the required environment variables:

   ### Flask Settings
   - `FLASK_ENV`: Application environment (development, production)
   - `FLASK_APP`: Main application file (app.py)
   - `FLASK_DEBUG`: Enable debug mode (1 for enabled, 0 for disabled)
   - `SECRET_KEY`: Secret key for Flask sessions and security
   - `PORT`: Port to run the application on (8080 recommended)

   ### PostgreSQL Database Settings
   - `POSTGRES_USER`: PostgreSQL username
   - `POSTGRES_PASSWORD`: PostgreSQL password
   - `POSTGRES_DB`: PostgreSQL database name
   - `DATABASE_URL`: Full PostgreSQL connection string (postgresql://username:password@host:port/dbname)

   ### Application Settings
   - `APP_VERSION`: Application version number
   - `MODEL_VERSION`: ML model version number
   - `CHECKPOINT_PATH`: Path to the model checkpoint file
   - `POKEMON_DATA_PATH`: Path to the Pokemon data CSV file
   - `LOG_FILE`: Path to the log file

## Quick Start with Docker

The easiest way to run the application is with Docker Compose:

```bash
# Clone the repository
git clone https://github.com/yourusername/pokemon-generator.git
cd pokemon-generator

# Create and configure .env file
# Start the application with Docker Compose
docker-compose up --build
```

The application will be available at http://localhost:5001

## Quick Start without Docker

For local development without Docker, you can use the local SQLite mode:

```bash
# Clone the repository
git clone https://github.com/yourusername/pokemon-generator.git
cd pokemon-generator

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application with SQLite database
./run_local.py
```

The application will be available at http://localhost:5001

## Example .env File

```
# Flask settings
FLASK_ENV=development
FLASK_APP=app.py
FLASK_DEBUG=1
SECRET_KEY=your_secure_secret_key
PORT=8080

# PostgreSQL database settings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=pokemon_dev
DATABASE_URL=postgresql://postgres:your_password@db:5432/pokemon_dev

# Application settings
APP_VERSION=1.2.0
MODEL_VERSION=2.0.0
CHECKPOINT_PATH=models/checkpoint.pth
POKEMON_DATA_PATH=data/Pokemon_stats.csv

# Log settings
LOG_FILE=logs/pokemon_generator.log
```

## Manual Installation

If you prefer to run the application without Docker:

1. Ensure you have Python 3.9+ installed
2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the database:
   ```
   python scripts/db_migrate.py
   ```

5. Run the application:
   ```
   python main.py
   ```

## Database

The application uses PostgreSQL. The schema is created from a `.template` file using environment variables. Ensure `.env` file is present and correctly configured.

## Challenges

- **GAN Training Stability**: Training GANs for high-resolution images (256x256) was challenging and often errored out, so I needed to constantly monitor and tune hyperparameters and regularization.
- **Experience**: I haven't had much experience with machine learning, much less with neural networks, so it was difficult to start. Documentation and AI helped a lot in learning how to create the GAN architecture and debug.
- **Stat Prediction Generalization**: Predicting reasonable stats for type combinations relies heavily on the RandomForest models, so I struggled to create accurate predictions. I had to fine-tune a lot.

## Future Improvements

- **Better GAN Architectures**: Experiment with more advanced GAN variants (e.g., StyleGAN, BigGAN) for improved image quality.
- **User Customization**: Gather data on more features/traits to allow users to further customize generated Pokemon or make more specific (e.g., color palette, body shape, accessories).
- **Interactive Training**: Enable users to provide feedback on generated images to guide future training (reinforcement learning or active learning).
- **API Endpoints**: Add more RESTful API endpoints for programmatic access to image generation and stat prediction.
- **Live Training Visualization**: Integrate live charts for loss curves and sample generations during training.

## Acknowledgements

- Pokemon data from PokeAPI
- PyTorch for deep learning framework
- Flask for web framework
- Tailwind for frontend design

