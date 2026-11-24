import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from flask import Flask, jsonify, request
from transformers import (RobertaForSequenceClassification, RobertaModel,
                          RobertaTokenizer)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set up paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "assets" / "model"

# Verify model directory exists
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model directory not found at {MODEL_DIR}")

# Define all model file paths
MODEL_FILES = {
    "vae": MODEL_DIR / "vae.pt",
    "classifier": MODEL_DIR / "best_model.pt",
    "scaler": MODEL_DIR / "scaler.pkl",
    "pca": MODEL_DIR / "pca.pkl",
    "selector": MODEL_DIR / "feature_selector.pkl",
    "emotion_model": MODEL_DIR / "best_emotion_model.pt",
    "sentiment_model": MODEL_DIR / "best_sentiment_model.pt",
    "num_classes": MODEL_DIR / "num_classes.txt",
    "input_dim": MODEL_DIR / "input_dim.txt",
    "feature_metadata": MODEL_DIR / "feature_metadata.txt"
}

# Verify all model files exist
missing_files = [name for name, path in MODEL_FILES.items() if not path.exists()]
if missing_files:
    raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")

# Load configuration
config = {
    'models': {
        'emotion': 'SamLowe/roberta-base-go_emotions',
        'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment',
        'embedding': 'roberta-base'
    },
    'max_len': 128,
    'emotion_labels': [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride',
        'realization', 'relief', 'remorse', 'sadness', 'surprise'
    ],
    'sentiment_labels': ['negative', 'neutral', 'positive']
}

# Load all models and processors at startup
logger.info(f"Loading models from {MODEL_DIR}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnhancedVAE(nn.Module):
    def __init__(self, input_dim=768, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class EnhancedDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5)
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.4)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.3)
        )
        self.skip1 = nn.Linear(input_dim, 256)
        self.skip2 = nn.Linear(512, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.input_bn(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        skip1 = self.skip1(x)
        x2 = x2 + skip1[:, :256]
        x3 = self.block3(x2)
        skip2 = self.skip2(x1)
        x3 = x3 + skip2[:, :128]
        return self.output(x3)

# Load feature metadata
def load_feature_metadata():
    feature_metadata = {
        'emotion': {},
        'sentiment': {},
        'latent': {}
    }
    
    try:
        with open(MODEL_FILES["feature_metadata"], 'r') as f:
            current_section = None
            for line in f:
                line = line.strip()
                if line.startswith('==='):
                    if 'Emotion' in line:
                        current_section = 'emotion'
                    elif 'Sentiment' in line:
                        current_section = 'sentiment'
                    elif 'Latent' in line:
                        current_section = 'latent'
                elif current_section and ':' in line:
                    key, desc = line.split(':', 1)
                    key = key.strip()
                    desc = desc.strip()
                    feature_metadata[current_section][key] = desc
    except Exception as e:
        logger.warning(f"Could not load feature metadata: {str(e)}")
    
    return feature_metadata

try:
    # Load models with proper initialization
    # 1. Load VAE
    vae_state_dict = torch.load(str(MODEL_FILES["vae"]), map_location=device)
    vae = EnhancedVAE(input_dim=768).to(device)
    vae.load_state_dict(vae_state_dict)
    vae.eval()

    # 2. Load classifier
    classifier_state_dict = torch.load(str(MODEL_FILES["classifier"]), map_location=device)
    with open(MODEL_FILES["input_dim"]) as f:
        input_dim = int(f.read().strip())
    with open(MODEL_FILES["num_classes"]) as f:
        num_classes = int(f.read().strip())
    model = EnhancedDNN(input_dim, num_classes).to(device)
    model.load_state_dict(classifier_state_dict)
    model.eval()

    # 3. Load feature processors
    scaler = joblib.load(str(MODEL_FILES["scaler"]))
    pca = joblib.load(str(MODEL_FILES["pca"]))
    selector = joblib.load(str(MODEL_FILES["selector"]))
    
    # 4. Load feature extractors with fine-tuned models
    feature_extractors = {
        'emotion': RobertaForSequenceClassification.from_pretrained(config['models']['emotion']).to(device).eval(),
        'sentiment': RobertaForSequenceClassification.from_pretrained(config['models']['sentiment']).to(device).eval(),
        'embedding': RobertaModel.from_pretrained(config['models']['embedding']).to(device).eval()
    }
    
    # Load fine-tuned weights for emotion and sentiment models
    feature_extractors['emotion'].load_state_dict(torch.load(str(MODEL_FILES["emotion_model"]), map_location=device))
    feature_extractors['sentiment'].load_state_dict(torch.load(str(MODEL_FILES["sentiment_model"]), map_location=device))
    
    tokenizers = {
        'emotion': RobertaTokenizer.from_pretrained(config['models']['emotion']),
        'sentiment': RobertaTokenizer.from_pretrained(config['models']['sentiment']),
        'embedding': RobertaTokenizer.from_pretrained(config['models']['embedding'])
    }
    
    # Load feature metadata
    feature_metadata = load_feature_metadata()
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    
    # Define feature names based on model configuration
    emotion_cols = [f'emotion_{i}' for i in range(len(config['emotion_labels']))]
    sentiment_col = ['sentiment_score']
    latent_cols = [f'latent_{i}' for i in range(128)]  # Assuming latent_dim=128
    
    logger.info("All models loaded successfully!")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def enhanced_preprocess(text: str) -> str:
    """Enhanced preprocessing matching the training pipeline"""
    patterns = [
        (re.compile(r"http\S+|www\S+|https\S+"), ''),
        (re.compile(r'@\w+'), 'USER'),
        (re.compile(r'#'), ''),
        (re.compile(r'[^\w\s]'), ''),
        (re.compile(r'\b(?:xa0|â|nbsp|ï|½|œ|"|"|�)+\b', flags=re.IGNORECASE), ''),
        (re.compile(r'\d+'), 'NUM'),
        (re.compile(r'\s+'), ' '),
        (re.compile(r'\b(?:fuck|shit|asshole|bitch|cunt|nigger|fag)\b', flags=re.IGNORECASE), 'SWEAR')
    ]
    text = text.lower().strip()
    for pattern, repl in patterns:
        text = pattern.sub(repl, text)
    return text

def extract_features(text: str) -> Dict[str, np.ndarray]:
    """Feature extraction identical to training"""
    start_time = time.time()
    text = enhanced_preprocess(text)
    features = {}
    
    for name, model in feature_extractors.items():
        inputs = tokenizers[name](text, return_tensors='pt', truncation=True, max_length=config['max_len']).to(device)
        
        with torch.no_grad():
            if name in ['emotion', 'sentiment']:
                outputs = model(**inputs)
                features[name] = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            else:  # Embeddings
                outputs = model(**inputs, output_hidden_states=True)
                last_layers = torch.stack(outputs.hidden_states[-4:])
                mean_pooled = last_layers.mean(dim=0).mean(dim=1)
                features['embeddings'] = mean_pooled.cpu().numpy()[0]
    
    # Calculate sentiment score exactly as in training
    features['sentiment_score'] = features['sentiment'][2] - features['sentiment'][0]
    logger.debug(f"Feature extraction completed in {time.time() - start_time:.2f}s")
    return features

def validate_feature_dimensions(features: Dict[str, np.ndarray]) -> bool:
    """Validate that extracted features match expected dimensions"""
    expected_dims = {
        'emotion': len(config['emotion_labels']),
        'sentiment': 3,  # negative, neutral, positive
        'embeddings': 768  # RoBERTa base embedding size
    }
    
    for feature_name, expected_dim in expected_dims.items():
        if feature_name not in features:
            logger.error(f"Missing feature: {feature_name}")
            return False
        if features[feature_name].shape[0] != expected_dim:
            logger.error(f"Dimension mismatch for {feature_name}. Expected {expected_dim}, got {features[feature_name].shape[0]}")
            return False
    return True

def get_feature_description(feature_name: str) -> str:
    """Get human-readable description of a feature"""
    if feature_name.startswith('emotion_'):
        idx = int(feature_name.split('_')[1])
        emotion_name = config['emotion_labels'][idx]
        return feature_metadata['emotion'].get(feature_name, f"{emotion_name} emotion")
    elif feature_name == 'sentiment_score':
        return "Overall sentiment (positive - negative)"
    elif feature_name.startswith('latent_'):
        idx = feature_name.split('_')[-1]
        return feature_metadata['latent'].get(f'latent_{idx}', f"Latent feature {idx}")
    return feature_name

def explain_prediction(X_scaled: np.ndarray) -> List[Dict[str, Union[str, float]]]:
    """Use Integrated Gradients to explain the prediction"""
    try:
        input_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_class = outputs.argmax().item()
        
        # Compute attributions
        attributions = ig.attribute(input_tensor, target=pred_class, return_convergence_delta=False)
        attributions = attributions.squeeze().cpu().detach().numpy()
        
        # Get top 2 most influential features
        top_indices = np.argsort(np.abs(attributions))[-2:][::-1]
        explanations = []
        
        for idx in top_indices:
            # Determine feature type based on index
            if idx < len(emotion_cols):
                feature_name = emotion_cols[idx]
            elif idx == len(emotion_cols):
                feature_name = 'sentiment_score'
            else:
                feature_name = f"latent_{idx - len(emotion_cols) - 1}"
                
            explanations.append({
                'feature': feature_name,
                'description': get_feature_description(feature_name),
                'impact': float(attributions[idx]),
                'direction': 'positive' if attributions[idx] > 0 else 'negative'
            })
        
        return explanations
    
    except Exception as e:
        logger.error(f"Error in explanation: {str(e)}")
        return []

def predict_text(text: str) -> Dict[str, Union[int, float, List[float], str, List[Dict[str, Union[str, float]]]]]:
    """Make predictions matching training pipeline with enhanced validation"""
    try:
        start_time = time.time()
        raw_features = extract_features(text)
        
        # Validate feature dimensions
        if not validate_feature_dimensions(raw_features):
            return {
                'error': 'Feature dimension mismatch',
                'status': 'error'
            }
            
        emotion_features = raw_features['emotion']
        sentiment_feature = np.array([raw_features['sentiment_score']])
        embedding_features = raw_features['embeddings']
        
        # Process through VAE exactly as in training
        with torch.no_grad():
            embeddings_tensor = torch.tensor(embedding_features, dtype=torch.float32).unsqueeze(0).to(device)
            mu, logvar = vae.encode(embeddings_tensor)
            latent_features = vae.reparameterize(mu, logvar).cpu().numpy()[0]
        
        # Combine features exactly as in training
        X = np.concatenate([emotion_features, sentiment_feature, latent_features]).reshape(1, -1)
        
        # Validate feature dimension before selection
        if X.shape[1] != selector.n_features_in_:
            return {
                'error': f'Feature dimension mismatch before selection. Expected {selector.n_features_in_}, got {X.shape[1]}',
                'status': 'error'
            }
        
        # Apply feature transformations in same order as training
        X_selected = selector.transform(X)
        
        # Split features into emotion/sentiment and latent components
        num_emotion_features = len(emotion_features) + 1  # +1 for sentiment score
        X_emotion = X_selected[:, :num_emotion_features]
        X_latent = X_selected[:, num_emotion_features:]
        
        # Apply PCA only to latent features
        X_latent_pca = pca.transform(X_latent)
        
        # Recombine features
        X_processed = np.concatenate([X_emotion, X_latent_pca], axis=1)
        
        # Standardize features
        X_scaled = scaler.transform(X_processed)
        
        # Validate final feature dimension
        if X_scaled.shape[1] != input_dim:
            return {
                'error': f'Final feature dimension mismatch. Expected {input_dim}, got {X_scaled.shape[1]}',
                'status': 'error'
            }
        
        # Make final prediction
        with torch.no_grad():
            input_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            output = model(input_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            prediction = int(np.argmax(probs))
            confidence = float(np.max(probs))
        
        # Get explanation
        explanations = explain_prediction(X_scaled)
        
        logger.info(f"Prediction completed in {time.time() - start_time:.2f}s")
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probs.tolist(),
            'explanations': explanations,
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return {
            'error': str(e),
            'status': 'error'
        }

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text in request', 'status': 'error'}), 400
        
    text = data['text']
    if not isinstance(text, str) or len(text.strip()) == 0:
        return jsonify({'error': 'Invalid text input', 'status': 'error'}), 400
    
    result = predict_text(text)
    if result.get('status') == 'error':
        return jsonify(result), 500
    return jsonify(result)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Missing texts array in request', 'status': 'error'}), 400
        
    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({'error': 'Invalid texts input', 'status': 'error'}), 400
    if len(texts) > 100:  # Limit batch size
        return jsonify({'error': 'Batch size too large (max 100)', 'status': 'error'}), 400
    
    results = [predict_text(text) for text in texts]
    return jsonify({'results': results})

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    """Legacy endpoint (maintained for backwards compatibility)"""
    if request.method == 'GET':
        return jsonify({
            'message': 'Send a POST request with text for detection',
            'status': 'info'
        })
    elif request.method == 'POST':
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text in request', 'status': 'error'}), 400
            
        text = data['text']
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({'error': 'Invalid text input', 'status': 'error'}), 400
        
        result = predict_text(text)
        if result.get('status') == 'error':
            return jsonify(result), 500
        return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    response = {
        'status': 'healthy',
        'models_loaded': True,
        'device': str(device),
        'num_classes': int(num_classes) if hasattr(num_classes, 'item') else num_classes,
        'input_dim': int(input_dim) if hasattr(input_dim, 'item') else input_dim,
    }
    
    # Add selector info if available
    if hasattr(selector, 'n_features_in_'):
        response['feature_selector_features'] = int(selector.n_features_in_)
    else:
        response['feature_selector_features'] = None
    
    # Add PCA info if available
    if hasattr(pca, 'n_components_'):
        response['pca_components'] = int(pca.n_components_)
    else:
        response['pca_components'] = None
    
    return jsonify(response)

def main():
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()