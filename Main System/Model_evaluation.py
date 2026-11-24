import re
import warnings
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (RobertaForSequenceClassification, RobertaModel,
                          RobertaTokenizer, logging)

warnings.filterwarnings('ignore')

# Suppress unnecessary warnings
logging.set_verbosity_error()

# --------------------- Configuration ---------------------
class EvalConfig:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = 42
        # Models for different feature types
        self.model_config = {
            'emotion': {
                'model_name': 'SamLowe/roberta-base-go_emotions',
                'feature_dim': 28,
                'model_path': 'best_emotion_model.pt',
                'labels': [
                    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                    'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride',
                    'realization', 'relief', 'remorse', 'sadness', 'surprise'
                ]
            },
            'sentiment': {
                'model_name': 'cardiffnlp/twitter-roberta-base-sentiment',
                'feature_dim': 3,
                'model_path': 'best_sentiment_model.pt',
                'labels': ['negative', 'neutral', 'positive']
            },
            'embedding': {
                'model_name': 'roberta-base',
                'feature_dim': 768,
                'model_path': 'best_embedding_model.pt'
            }
        }
        self.max_len = 128
        self.batch_size = 32
        self.latent_dim = 64

        # Bullying type configuration
        self.clean_bullying_types = ['not_cyberbullying', 'ethnicity', 'religion', 'gender']
        self.bullying_descriptions = {
            'not_cyberbullying': 'Normal/Non-bullying content',
            'ethnicity': 'Ethnicity/Race-based bullying',
            'religion': 'Religion-based bullying',
            'gender': 'Gender/Sexual orientation bullying'
        }

eval_config = EvalConfig()

# --------------------- Enhanced DNN Classifier ---------------------
class EnhancedDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5))
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.4))
        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Mish(),
            nn.Dropout(0.3))
        self.skip1 = nn.Linear(input_dim, 256)
        self.skip2 = nn.Linear(512, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.input_bn(x)
        h1 = self.block1(x)
        h2 = self.block2(h1) + F.silu(self.skip1(x))
        h3 = self.block3(h2) + F.silu(self.skip2(h1))
        return self.output(h3)

# --------------------- Flexible VAE ---------------------
class FlexibleVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
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

# --------------------- Enhanced Preprocessing ---------------------
def enhanced_preprocess(text):
    """Enhanced preprocessing with better handling of context-rich elements"""
    patterns = [
        (re.compile(r"http\S+|www\S+|https\S+"), ''),
        (re.compile(r'@\w+'), 'USER'),
        (re.compile(r'#(\w+)'), r'HASHTAG_\1'),
        (re.compile(r'[^\w\s]'), ' '),
        (re.compile(r'\b(?:xa0|â|nbsp|ï|½|œ|"|"|�)+\b', flags=re.IGNORECASE), ''),
        (re.compile(r'\d+'), 'NUM'),
        (re.compile(r'\s+'), ' '),
        (re.compile(r'\b(?:fuck|shit|asshole)\b', flags=re.IGNORECASE), 'SWEAR_MILD'),
        (re.compile(r'\b(?:bitch|cunt|fag)\b', flags=re.IGNORECASE), 'SWEAR_SEVERE'),
        (re.compile(r'\b(?:nigger|retard)\b', flags=re.IGNORECASE), 'SLUR_RACIST'),
    ]
    text = text.lower().strip()
    for pattern, repl in patterns:
        text = pattern.sub(repl, text)
    return text.strip()

# --------------------- Robust Feature Extraction ---------------------
class RobustFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.tokenizers = {}

        for feature_type, cfg in config.model_config.items():
            print(f"Loading {feature_type} model...")
            try:
                if feature_type in ['emotion', 'sentiment']:
                    # Load base model first
                    model = RobertaForSequenceClassification.from_pretrained(
                        cfg['model_name'], num_labels=cfg['feature_dim']
                    ).to(config.device)
                    
                    # Load fine-tuned weights with key mapping
                    state_dict = self._load_state_dict_with_key_mapping(cfg['model_path'])
                    
                    # Apply state dict with strict=False to handle key mismatches
                    model.load_state_dict(state_dict, strict=False)
                    self.models[feature_type] = model
                    print(f"✅ Loaded fine-tuned {feature_type} model with key mapping")

                else:  # embedding model
                    # Load base model
                    model = RobertaModel.from_pretrained(cfg['model_name']).to(config.device)
                    
                    # Load fine-tuned weights with key mapping
                    state_dict = self._load_state_dict_with_key_mapping(cfg['model_path'])
                    
                    # Apply state dict
                    model.load_state_dict(state_dict, strict=False)
                    self.models[feature_type] = model
                    print(f"✅ Loaded fine-tuned embedding model with key mapping")

                self.tokenizers[feature_type] = RobertaTokenizer.from_pretrained(cfg['model_name'])

            except Exception as e:
                print(f"Error loading {feature_type} model: {e}")
                # Fallback to base models
                if feature_type in ['emotion', 'sentiment']:
                    self.models[feature_type] = RobertaForSequenceClassification.from_pretrained(
                        cfg['model_name'], num_labels=cfg['feature_dim']
                    ).to(config.device)
                    print(f"⚠️ Using base {feature_type} model as fallback")
                else:
                    self.models[feature_type] = RobertaModel.from_pretrained(
                        cfg['model_name']
                    ).to(config.device)
                    print(f"⚠️ Using base embedding model as fallback")
                self.tokenizers[feature_type] = RobertaTokenizer.from_pretrained(cfg['model_name'])

    def _load_state_dict_with_key_mapping(self, model_path):
        """Load state dict and map keys from base_model.* to standard format"""
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        
        # Handle DataParallel wrappers
        if hasattr(state_dict, 'module'):
            state_dict = state_dict.module.state_dict()
        
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Map base_model. keys to standard keys
            if key.startswith('base_model.'):
                new_key = key.replace('base_model.', '')
            elif key.startswith('roberta.'):
                # Already in standard format
                new_key = key
            elif key.startswith('module.base_model.'):
                new_key = key.replace('module.base_model.', '')
            elif key.startswith('module.roberta.'):
                new_key = key.replace('module.roberta.', 'roberta.')
            else:
                new_key = key
            
            # Handle classifier layer mappings
            if 'classifier' in new_key and 'emotion_head' not in new_key and 'sentiment_head' not in new_key:
                # Keep classifier as is for sequence classification models
                pass
            elif 'emotion_head' in new_key or 'sentiment_head' in new_key:
                # These are custom heads, keep them
                pass
            elif 'embedding_head' in new_key:
                # For embedding model custom head
                pass
            
            new_state_dict[new_key] = value
        
        return new_state_dict

    def extract_features(self, texts, batch_size=32):
        features = {'emotion': [], 'sentiment': [], 'embeddings': []}

        for feature_type in ['emotion', 'sentiment', 'embedding']:
            model = self.models[feature_type]
            tokenizer = self.tokenizers[feature_type]
            model.eval()

            for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {feature_type}"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(
                    batch, return_tensors='pt', padding=True,
                    truncation=True, max_length=eval_config.max_len
                ).to(eval_config.device)

                with torch.no_grad():
                    if feature_type in ['emotion', 'sentiment']:
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)
                        features[feature_type].append(probs.cpu())
                    else:
                        # For embedding model - use the same pooling strategy as training
                        outputs = model(**inputs, output_hidden_states=True)

                        # Use pooling strategy - this might be where the dimension difference comes from
                        last_layers = torch.stack(outputs.hidden_states[-4:])
                        mean_embeddings = last_layers.mean(dim=0)
                        cls_embedding = mean_embeddings[:, 0, :]
                        mean_pooled = mean_embeddings.mean(dim=1)
                        
                        # Check if we need to concatenate or use different pooling
                        # Try different combinations to match the training dimension
                        combined_embedding = torch.cat([cls_embedding, mean_pooled], dim=1)
                        features['embeddings'].append(combined_embedding.cpu())

        # Combine features
        combined_features = {
            'emotion': torch.cat(features['emotion']).numpy(),
            'sentiment': torch.cat(features['sentiment']).numpy(),
            'embeddings': torch.cat(features['embeddings']).numpy()
        }

        # Enhanced sentiment scoring
        sentiment_scores = combined_features['sentiment'][:, 2] - combined_features['sentiment'][:, 0]
        sentiment_confidence = np.max(combined_features['sentiment'], axis=1)

        return {
            'emotion': combined_features['emotion'],
            'sentiment_score': sentiment_scores,
            'sentiment_confidence': sentiment_confidence,
            'sentiment_raw': combined_features['sentiment'],
            'embeddings': combined_features['embeddings']
        }

# --------------------- Enhanced Evaluation ---------------------
def comprehensive_evaluation(model, X_test, y_test, class_names, label_encoder):
    """Comprehensive model evaluation with detailed metrics"""
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    y_true = y_test.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_probs = probs.cpu().numpy()

    # Comprehensive metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION ON UNSEEN DATA")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Balanced Accuracy: {report['macro avg']['recall']:.4f}")

    # Per-class metrics
    print(f"\n📊 Per-Class Performance:")
    for i, class_name in enumerate(class_names):
        class_report = report[class_name]
        print(f"  {class_name}: Precision={class_report['precision']:.3f}, "
              f"Recall={class_report['recall']:.3f}, F1={class_report['f1-score']:.3f}")

    # Enhanced confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Unseen Data Evaluation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('unseen_data_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Probability distribution per class
    plt.figure(figsize=(15, 10))
    n_classes = len(class_names)

    for i, class_name in enumerate(class_names):
        plt.subplot(2, 2, i+1)
        plt.hist(y_probs[y_true == i, i], bins=20, alpha=0.7,
                label=f'True {class_name}', color=f'C{i}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title(f'Probability Distribution - {class_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('unseen_data_probability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_probs,
        'label_encoder': label_encoder
    }

# --------------------- Data Loading for Evaluation ---------------------
def load_evaluation_data(filepath):
    """Load and preprocess evaluation data"""
    df = pd.read_csv(filepath)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must contain 'text' and 'label' columns")

    print("📊 Evaluation Data Distribution:")
    print(df['label'].value_counts())

    # Clean data and standardize labels
    df = df[['text', 'label']].dropna()
    df['cleaned_text'] = df['text'].astype(str).apply(enhanced_preprocess)
    df['cleaned_text'] = df['cleaned_text'].str.strip()
    df = df[df['cleaned_text'].str.len() > 1]
    df = df[df['cleaned_text'].str.contains('[a-zA-Z]')]

    # Standardize labels
    label_mapping = {
        'not_cyberbullying': 'not_cyberbullying',
        'ethnicity/race': 'ethnicity',
        'ethnicity': 'ethnicity',
        'religion': 'religion',
        'gender': 'gender',
        'gender/sexual': 'gender'
    }

    df['label_clean'] = df['label'].map(label_mapping)
    df = df.dropna(subset=['label_clean'])

    print("📊 Cleaned Evaluation Data Distribution:")
    print(df['label_clean'].value_counts())

    return df

# --------------------- Get VAE Input Dimension ---------------------
def get_vae_input_dim(vae_state_dict):
    """Extract the input dimension from VAE state dict"""
    # The encoder's first layer weight shape should be [hidden_dim, input_dim]
    encoder_weight_shape = vae_state_dict['encoder.0.weight'].shape
    return encoder_weight_shape[1]  # input dimension

# --------------------- Alternative: Skip VAE and use direct features ---------------------
def evaluate_without_vae(classifier, feature_extractor, eval_df, scaler, label_encoder, class_names):
    """Evaluate using only emotion and sentiment features, skipping VAE"""
    print("Using alternative evaluation without VAE...")
    
    # Extract features
    eval_features = feature_extractor.extract_features(eval_df['cleaned_text'].tolist())
    
    # Prepare features without VAE latent space
    # Use only emotion and sentiment features
    X_eval = np.concatenate([
        eval_features['emotion'],
        eval_features['sentiment_score'].reshape(-1, 1),
        eval_features['sentiment_raw']
    ], axis=1)
    
    print(f"📊 Alternative Features Shape: {X_eval.shape}")
    
    # Standardize features
    X_eval_final = scaler.transform(X_eval)
    
    # Encode labels
    y_eval = label_encoder.transform(eval_df['label_clean'].values)
    
    # Convert to tensors
    X_eval_t = torch.tensor(X_eval_final, dtype=torch.float32).to(eval_config.device)
    y_eval_t = torch.tensor(y_eval, dtype=torch.long).to(eval_config.device)
    
    # Evaluate
    return comprehensive_evaluation(classifier, X_eval_t, y_eval_t, class_names, label_encoder)

# --------------------- Main Evaluation Pipeline ---------------------
def evaluate_on_unseen_data():
    print(f"Using device: {eval_config.device}")
    print("🔍 EVALUATING ON UNSEEN DATASET 2.csv")

    try:
        # Load evaluation data
        print("Loading evaluation data...")
        eval_df = load_evaluation_data("2.csv")

        # Load saved artifacts
        print("Loading saved models and artifacts...")
        
        # Load model info to get feature dimensions
        model_info = joblib.load('model_info.pkl')
        feature_info = joblib.load('feature_info.pkl')
        
        # Load label encoder
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Load scaler
        scaler = joblib.load('scaler.pkl')
        
        # Try to load VAE with correct dimensions
        vae_state_dict = torch.load('vae_model.pt', map_location=eval_config.device)
        vae_input_dim = get_vae_input_dim(vae_state_dict)
        print(f"Detected VAE input dimension: {vae_input_dim}")
        
        vae = FlexibleVAE(input_dim=vae_input_dim, latent_dim=eval_config.latent_dim).to(eval_config.device)
        vae.load_state_dict(vae_state_dict)
        vae.eval()
        print("✅ VAE loaded successfully")
        
        # Load main classifier
        classifier_state_dict = torch.load('bullying_classifier_model.pt', map_location=eval_config.device)
        classifier = EnhancedDNN(
            input_dim=model_info['input_dim'], 
            num_classes=model_info['num_classes']
        ).to(eval_config.device)
        classifier.load_state_dict(classifier_state_dict)
        classifier.eval()
        print("✅ Classifier loaded successfully")

        # Initialize feature extractor
        feature_extractor = RobustFeatureExtractor(eval_config)

        # Extract features for evaluation data
        print("\nExtracting features for evaluation set...")
        eval_features = feature_extractor.extract_features(eval_df['cleaned_text'].tolist())

        print(f"Embedding features shape: {eval_features['embeddings'].shape}")
        
        # Check if embedding dimension matches VAE input dimension
        if eval_features['embeddings'].shape[1] != vae_input_dim:
            print(f"⚠️ Dimension mismatch: embeddings have {eval_features['embeddings'].shape[1]} features, VAE expects {vae_input_dim}")
            print("🔄 Trying alternative embedding pooling...")
            
            # Try different pooling strategy to match dimensions
            feature_extractor_alt = RobustFeatureExtractor(eval_config)
            
            # Re-extract with different pooling
            for feature_type in ['embedding']:
                model = feature_extractor_alt.models[feature_type]
                tokenizer = feature_extractor_alt.tokenizers[feature_type]
                model.eval()
                
                alt_embeddings = []
                for i in tqdm(range(0, len(eval_df), eval_config.batch_size), desc="Re-extracting embeddings"):
                    batch = eval_df['cleaned_text'].tolist()[i:i+eval_config.batch_size]
                    inputs = tokenizer(
                        batch, return_tensors='pt', padding=True,
                        truncation=True, max_length=eval_config.max_len
                    ).to(eval_config.device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        # Try using only the last layer's CLS token
                        last_hidden = outputs.hidden_states[-1]
                        cls_embedding = last_hidden[:, 0, :]
                        alt_embeddings.append(cls_embedding.cpu())
                
                eval_features['embeddings'] = torch.cat(alt_embeddings).numpy()
                print(f"New embedding features shape: {eval_features['embeddings'].shape}")

        # If dimensions still don't match, use alternative evaluation without VAE
        if eval_features['embeddings'].shape[1] != vae_input_dim:
            print("❌ Still dimension mismatch. Using evaluation without VAE...")
            eval_results = evaluate_without_vae(
                classifier, feature_extractor, eval_df, scaler, label_encoder, eval_config.clean_bullying_types
            )
        else:
            # Transform embeddings using trained VAE
            with torch.no_grad():
                eval_embeddings_t = torch.tensor(eval_features['embeddings'], dtype=torch.float32).to(eval_config.device)
                eval_latent, _ = vae.encode(eval_embeddings_t)
                eval_latent = eval_latent.cpu().numpy()

            # Prepare feature matrix
            def prepare_features(emotion, sentiment_score, latent, sentiment_raw=None):
                base_features = np.concatenate([emotion, sentiment_score.reshape(-1, 1)], axis=1)
                if sentiment_raw is not None:
                    base_features = np.concatenate([base_features, sentiment_raw], axis=1)
                return np.concatenate([base_features, latent], axis=1)

            X_eval = prepare_features(
                eval_features['emotion'],
                eval_features['sentiment_score'],
                eval_latent,
                eval_features['sentiment_raw']
            )

            print(f"📊 Evaluation Features Shape: {X_eval.shape}")

            # Standardize features using the saved scaler
            X_eval_final = scaler.transform(X_eval)

            # Encode labels
            y_eval = label_encoder.transform(eval_df['label_clean'].values)

            # Convert to tensors
            X_eval_t = torch.tensor(X_eval_final, dtype=torch.float32).to(eval_config.device)
            y_eval_t = torch.tensor(y_eval, dtype=torch.long).to(eval_config.device)

            # Comprehensive evaluation
            eval_results = comprehensive_evaluation(classifier, X_eval_t, y_eval_t, eval_config.clean_bullying_types, label_encoder)

        # Final results
        print(f"\n🎯 FINAL EVALUATION RESULTS ON UNSEEN DATA (2.csv):")
        print(f"Dataset size: {len(eval_df)} samples")
        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print(f"Macro F1: {eval_results['macro_f1']:.4f}")
        print(f"Weighted F1: {eval_results['weighted_f1']:.4f}")
        print(f"Precision: {eval_results['precision']:.4f}")
        print(f"Recall: {eval_results['recall']:.4f}")

        # Detailed per-class analysis
        print(f"\n📊 DETAILED PER-CLASS PERFORMANCE:")
        report = eval_results['classification_report']
        for class_name in eval_config.clean_bullying_types:
            if class_name in report:
                class_metrics = report[class_name]
                print(f"\n{class_name.upper()}:")
                print(f"  Precision: {class_metrics['precision']:.3f}")
                print(f"  Recall:    {class_metrics['recall']:.3f}")
                print(f"  F1-Score:  {class_metrics['f1-score']:.3f}")
                print(f"  Support:   {class_metrics['support']}")

        # Class distribution analysis
        print(f"\n📈 CLASS DISTRIBUTION ANALYSIS:")
        class_distribution = eval_df['label_clean'].value_counts()
        for class_name in eval_config.clean_bullying_types:
            count = class_distribution.get(class_name, 0)
            percentage = (count / len(eval_df)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

        # Save evaluation results
        evaluation_summary = {
            'dataset': '2.csv',
            'dataset_size': len(eval_df),
            'class_distribution': class_distribution.to_dict(),
            'overall_metrics': {
                'accuracy': eval_results['accuracy'],
                'macro_f1': eval_results['macro_f1'],
                'weighted_f1': eval_results['weighted_f1'],
                'precision': eval_results['precision'],
                'recall': eval_results['recall']
            },
            'per_class_metrics': {class_name: report[class_name] for class_name in eval_config.clean_bullying_types if class_name in report},
            'confusion_matrix': eval_results['confusion_matrix'].tolist(),
            'predictions': eval_results['predictions'],
            'true_labels': y_eval
        }

        joblib.dump(evaluation_summary, 'unseen_data_evaluation_results.pkl')
        print("\n✅ Evaluation results saved to 'unseen_data_evaluation_results.pkl'")

        print("\n✅ Evaluation completed successfully!")
        print("📁 Generated files:")
        print("  - unseen_data_confusion_matrix.png")
        print("  - unseen_data_probability_analysis.png")
        print("  - unseen_data_evaluation_results.pkl")

        return evaluation_summary

    except Exception as e:
        print(f"Error in evaluation pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    evaluate_on_unseen_data()