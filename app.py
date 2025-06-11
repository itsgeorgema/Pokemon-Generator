import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
import random
from src.models.CreateImage import generate_and_save_image
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta, timezone
import threading
import time
from src.config.config import config, Config
from sqlalchemy import text
import uuid
import numpy as np


load_dotenv()

app = Flask(__name__)
app_config = config[os.getenv('FLASK_ENV', 'default')]
app.config.from_object(app_config)

# Validate environment variables
try:
    Config.validate_env_vars()
except EnvironmentError as e:
    raise

# Handle Render's postgres:// prefix if present
if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)

if os.getenv('DOCKER_CONTAINER') == 'true':
    if 'localhost' in app.config['SQLALCHEMY_DATABASE_URI']:
        app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('localhost', 'db')
    db_url = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url

# Initialize database
try:
    db = SQLAlchemy(app)
    with app.app_context():
        db.session.execute(text("SELECT 1"))
        db.session.commit()
    print("[INFO] Successfully connected to the database.")
except Exception as e:
    print(f"[ERROR] Database connection failed: {str(e)}")
    raise

# Define database model for temporary images
class GeneratedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False, unique=True)
    image_data = db.Column(db.LargeBinary, nullable=False)  # Store the actual image data
    content_type = db.Column(db.String(100), default='image/png')  # MIME type
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    expires_at = db.Column(db.DateTime, nullable=False)
    type1 = db.Column(db.String(50), nullable=False)  # Store Pokemon type for stats
    type2 = db.Column(db.String(50), nullable=True)   # Secondary type may be None

    def __init__(self, filename, image_data, type1, type2=None):
        self.filename = filename
        self.image_data = image_data
        self.type1 = type1
        self.type2 = type2
        # Ensure timezone-aware datetime for expires_at
        self.expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)  # Images expire after 10 minutes

# Create database tables
with app.app_context():
    db.create_all()

# Load and preprocess Pokemon data
def load_pokemon_data():
    try:
        metadata = pd.read_csv(app.config['POKEMON_DATA_PATH'])
        
        type1_col = 'Type 1' if 'Type 1' in metadata.columns else 'Type1'
        type2_col = 'Type 2' if 'Type 2' in metadata.columns else 'Type2'
        
        types = sorted(list(set(metadata[type1_col].dropna().tolist() + metadata[type2_col].dropna().tolist())))
        
        type_encoder = LabelEncoder()
        metadata['Type1_encoded'] = type_encoder.fit_transform(metadata[type1_col])
        
        metadata['Type2_filled'] = metadata[type2_col].fillna('None')
        all_types = list(type_encoder.classes_) + ['None']
        type_encoder.classes_ = np.array(all_types)
        metadata['Type2_encoded'] = type_encoder.transform(metadata['Type2_filled'])
        
        stats_data = pd.read_csv(app.config['POKEMON_DATA_PATH'])
        
        models = {}
        for stat in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:
            # Create a mapping from Type 1/Type 2 to the stats
            type_to_stat = {}
            for _, row in stats_data.iterrows():
                type1 = row['Type 1']
                type2 = row['Type 2'] if pd.notna(row['Type 2']) else 'None'
                key = (type1, type2)
                if key not in type_to_stat:
                    type_to_stat[key] = []
                type_to_stat[key].append(row[stat])
            
            # For each type combination, take the average stat value
            for key, values in type_to_stat.items():
                type_to_stat[key] = sum(values) / len(values)
            
            # Train a model on the encoded types
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Create training data from the type_to_stat mapping
            X_train = []
            y_train = []
            for (type1, type2), stat_value in type_to_stat.items():
                type1_encoded = 0
                type2_encoded = 0
                
                if type1 in type_encoder.classes_:
                    type1_idx = np.where(type_encoder.classes_ == type1)[0]
                    if len(type1_idx) > 0:
                        type1_encoded = int(type1_idx[0])
                
                if type2 in type_encoder.classes_:
                    type2_idx = np.where(type_encoder.classes_ == type2)[0]
                    if len(type2_idx) > 0:
                        type2_encoded = int(type2_idx[0])
                
                X_train.append([type1_encoded, type2_encoded])
                y_train.append(stat_value)
            
            model.fit(X_train, y_train)
            models[stat] = model
            
        return metadata, types, type_encoder, models
    except Exception as e:
        raise

# Load data at startup
try:
    metadata, types, type_encoder, stat_models = load_pokemon_data()
except Exception as e:
    raise

def random_ability():
    try:
        if 'Abilities' in metadata.columns:
            all_abilities = [ability for sublist in metadata['Abilities'] for ability in sublist]
        elif 'abilities' in metadata.columns:
            all_abilities = [ability for sublist in metadata['abilities'] for ability in sublist]
        else:
            # If no abilities column is found, return a default list
            return random.choice(['Overgrow', 'Blaze', 'Torrent', 'Intimidate', 'Levitate', 
                                 'Static', 'Synchronize', 'Sand Veil', 'Immunity'])
        
        ability_counts = Counter(all_abilities)
        common_abilities = {ability: count for ability, count in ability_counts.items() if count >= 2}
        abilities_list = list(common_abilities.keys())
        weights = list(common_abilities.values())
        return random.choices(abilities_list, weights=weights, k=1)[0]
    except Exception as e:
        return random.choice(['Overgrow', 'Blaze', 'Torrent', 'Intimidate', 'Levitate', 
                             'Static', 'Synchronize', 'Sand Veil', 'Immunity'])

def predict_stats(type1, type2):
    try:
        type1_encoded = 0
        type2_encoded = 0
        
        if type1 in type_encoder.classes_:
            type1_idx = np.where(type_encoder.classes_ == type1)[0]
            if len(type1_idx) > 0:
                type1_encoded = int(type1_idx[0])
        
        type2_val = type2 if type2 else 'None'
        if type2_val in type_encoder.classes_:
            type2_idx = np.where(type_encoder.classes_ == type2_val)[0]
            if len(type2_idx) > 0:
                type2_encoded = int(type2_idx[0])
        
        stats = {}
        for stat, model in stat_models.items():
            predicted = model.predict([[type1_encoded, type2_encoded]])[0]
            stats[stat.lower().replace(' ', '_').replace('.', '')] = round(predicted)
            
        return stats
    except Exception as e:
        return {
            'hp': 80, 'attack': 80, 'defense': 80, 
            'sp_atk': 80, 'sp_def': 80, 'speed': 80
        }

def cleanup_expired_images():
    """Background task to clean up expired images"""
    while True:
        try:
            with app.app_context():
                cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=10)
                
                # Delete expired records from the database
                try:
                    # Use SQLAlchemy to delete expired records with explicit comparison
                    expired_records = GeneratedImage.query.filter(
                        text("expires_at < :cutoff_time")
                    ).params(cutoff_time=cutoff_time).all()
                    
                    for record in expired_records:
                        db.session.delete(record)
                    
                    db.session.commit()
                    
                    if expired_records:
                        pass
                except Exception as db_error:
                    pass
            
        except Exception as e:
            pass
        
        time.sleep(60)

def migrate_to_timezone_aware():
    """Migrate existing records to timezone-aware datetimes"""
    try:
        with app.app_context():
            images = GeneratedImage.query.all()
            updated = 0
            
            for image in images:
                try:
                    if image.expires_at and image.expires_at.tzinfo is None:
                        image.expires_at = image.expires_at.replace(tzinfo=timezone.utc)
                        updated += 1
                    
                    if image.created_at and image.created_at.tzinfo is None:
                        image.created_at = image.created_at.replace(tzinfo=timezone.utc)
                        updated += 1
                except Exception as inner_e:
                    pass
            
            if updated > 0:
                db.session.commit()
                pass
    except Exception as e:
        pass

with app.app_context():
    db.create_all()
    migrate_to_timezone_aware()

if not app.config.get('TESTING'):
    cleanup_thread = threading.Thread(target=cleanup_expired_images, daemon=True)
    cleanup_thread.start()

def cleanup_all_images():
    """Delete all previously generated images and database records"""
    try:
        # Delete all database records
        with app.app_context():
            try:
                records = GeneratedImage.query.all()
                for record in records:
                    db.session.delete(record)
                db.session.commit()
                if records:
                    pass
            except Exception as db_error:
                pass
    except Exception as e:
        pass

@app.route('/')
def index():
    return render_template('index.html', 
                           types=types, 
                           config={
                               'APP_VERSION': app.config.get('APP_VERSION', '1.0.0'),
                               'MODEL_VERSION': app.config.get('MODEL_VERSION', '2.0.0')
                           })

def generate_ivs():
    # Generate a random IV (0-31) for each stat
    return {
        'hp': random.randint(0, 31),
        'attack': random.randint(0, 31),
        'defense': random.randint(0, 31),
        'sp_atk': random.randint(0, 31),
        'sp_def': random.randint(0, 31),
        'speed': random.randint(0, 31)
    }

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        type1 = str(data.get('type1')).strip().capitalize()
        type2 = str(data.get('type2')).strip().capitalize() if data.get('type2') else None
        height = float(data.get('height', 1.0))
        weight = float(data.get('weight', 20.0))
        generation = int(data.get('generation', 1))
        legendary = bool(data.get('legendary', False))
        
        if not data.get('name'):
            return jsonify({"error": "Pokemon name is required"}), 400
        
        name = str(data.get('name')).strip()

        if not types or type1 not in types:
            return jsonify({"error": f"Invalid or missing primary type: {type1}"}), 400
        if type2 and type2 not in types:
            return jsonify({"error": f"Invalid secondary type: {type2}"}), 400

        stats = predict_stats(type1, type2)
        ivs = generate_ivs()
        for stat in stats:
            stats[stat] += ivs[stat]

        try:
            cleanup_all_images()
            # Generate image in memory
            image_bytes = generate_and_save_image(
                type1, type2, height, weight, generation, legendary,
                checkpoint_path=app.config['CHECKPOINT_PATH'],
                data_path=app.config['POKEMON_DATA_PATH']
            )
            filename = f"{uuid.uuid4().hex}.png"
            # Save image record to database with the binary data
            image_record = GeneratedImage(
                filename=filename,
                image_data=image_bytes,
                type1=type1,
                type2=type2
            )
            db.session.add(image_record)
            db.session.commit()
            pass
        except Exception as e:
            pass
            return jsonify({"error": "Failed to generate Pokemon image"}), 500

        # Return response with image URL and stats
        return jsonify({
            "type1": type1,
            "type2": type2,
            "name": name,
            "ability": random_ability(),
            "stats": stats,
            "ivs": ivs,
            "image_url": f"/image/{filename}?t={int(time.time())}"
        })

    except Exception as e:
        pass
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/image/<path:filename>')
def serve_generated_image(filename):
    """Serve a generated image from the database"""
    try:
        # Query the database for the image
        image_record = GeneratedImage.query.filter_by(filename=filename).first()
        
        if image_record and image_record.image_data:
            # Create a response with the image data
            response = make_response(image_record.image_data)
            response.headers.set('Content-Type', image_record.content_type)
            return response
        else:
            pass
            return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        pass
        return jsonify({"error": f"Error serving image: {str(e)}"}), 500

@app.route('/public/<path:filename>')
def serve_public_file(filename):
    """Serve files from the public directory"""
    try:
        public_dir = 'public'
        filepath = os.path.join(public_dir, filename)
        if os.path.exists(filepath):
            return send_from_directory(public_dir, filename)
        else:
            pass
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        pass
        return jsonify({"error": f"Error serving file: {str(e)}"}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    pass
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
