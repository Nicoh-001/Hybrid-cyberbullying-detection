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
import torch.optim as optim
from captum.attr import IntegratedGradients
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (RobertaForSequenceClassification, RobertaModel,
                          RobertaTokenizer, logging)

warnings.filterwarnings('ignore')

# Suppress unnecessary warnings
logging.set_verbosity_error()

# --------------------- Enhanced Configuration ---------------------
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = 42
        # Models - USING SENTIMENT AND EMBEDDING MODELS ONLY
        self.model_config = {
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
        self.classifier_epochs = 100
        self.classifier_lr = 3e-4
        self.vae_lr = 1e-4
        self.captum_samples = 50
        self.n_splits = 5

        # Bullying type configuration
        self.raw_bullying_types = ['not_cyberbullying', 'ethnicity/race', 'religion', 'gender/sexual']
        self.clean_bullying_types = ['not_cyberbullying', 'ethnicity', 'religion', 'gender']
        self.bullying_descriptions = {
            'not_cyberbullying': 'Normal/Non-bullying content',
            'ethnicity': 'Ethnicity/Race-based bullying',
            'religion': 'Religion-based bullying',
            'gender': 'Gender/Sexual orientation bullying'
        }

config = Config()

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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_bn(x)
        h1 = self.block1(x)
        h2 = self.block2(h1) + F.silu(self.skip1(x))
        h3 = self.block3(h2) + F.silu(self.skip2(h1))
        return self.output(h3)

# --------------------- Improved VAE ---------------------
class ImprovedVAE(nn.Module):
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

# --------------------- Robust Data Loading ---------------------
def load_and_balance_data(filepath, test_size=0.2, val_size=0.1):
    """Load data and ensure balanced distribution"""
    df = pd.read_csv(filepath)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must contain 'text' and 'label' columns")

    print("📊 Original Data Distribution:")
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
        'religion': 'religion',
        'gender/sexual': 'gender'
    }

    df['label_clean'] = df['label'].map(label_mapping)
    df = df.dropna(subset=['label_clean'])

    print("📊 Cleaned Data Distribution:")
    print(df['label_clean'].value_counts())

    # Balance the dataset
    not_bullying_df = df[df['label_clean'] == 'not_cyberbullying']
    bullying_df = df[df['label_clean'] != 'not_cyberbullying']

    print(f"\n📈 Class Balance Analysis:")
    print(f"Non-bullying samples: {len(not_bullying_df)}")
    print(f"Bullying samples: {len(bullying_df)}")
    print(f"Bullying type distribution:")
    print(bullying_df['label_clean'].value_counts())

    max_bullying_samples = len(bullying_df)
    not_bullying_sampled = not_bullying_df.sample(
        n=min(len(not_bullying_df), max_bullying_samples),
        random_state=config.random_state
    )

    balanced_df = pd.concat([not_bullying_sampled, bullying_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=config.random_state).reset_index(drop=True)

    print(f"\n⚖️ Balanced Dataset:")
    print(f"Total samples: {len(balanced_df)}")
    print(balanced_df['label_clean'].value_counts())

    # Encode labels
    label_encoder = LabelEncoder()
    balanced_df['label_encoded'] = label_encoder.fit_transform(balanced_df['label_clean'])

    # Split data with stratification
    train_val_df, test_df = train_test_split(
        balanced_df, test_size=test_size, stratify=balanced_df['label_clean'],
        random_state=config.random_state
    )

    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size/(1-test_size),
        stratify=train_val_df['label_clean'], random_state=config.random_state
    )

    print(f"\n📊 Final Split Summary:")
    for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        dist = split_df['label_clean'].value_counts()
        print(f"{split_name} ({len(split_df)} samples):")
        for label_type, count in dist.items():
            print(f"  {label_type}: {count} samples")
        print()

    return train_df, val_df, test_df, label_encoder

# --------------------- UPDATED: Robust Feature Extraction with Key Mapping ---------------------
class RobustFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.tokenizers = {}

        # LOAD SENTIMENT AND EMBEDDING MODELS ONLY
        for feature_type, cfg in config.model_config.items():
            print(f"Loading {feature_type} model...")
            try:
                if feature_type == 'sentiment':
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
                if feature_type == 'sentiment':
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
        features = {'sentiment': [], 'embeddings': []}

        # EXTRACT SENTIMENT AND EMBEDDINGS ONLY
        for feature_type in ['sentiment', 'embedding']:
            model = self.models[feature_type]
            tokenizer = self.tokenizers[feature_type]
            model.eval()

            for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {feature_type}"):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(
                    batch, return_tensors='pt', padding=True,
                    truncation=True, max_length=config.max_len
                ).to(config.device)

                with torch.no_grad():
                    if feature_type == 'sentiment':
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)
                        features['sentiment'].append(probs.cpu())
                    else:
                        # For embedding model
                        outputs = model(**inputs, output_hidden_states=True)

                        # Use pooling strategy
                        last_layers = torch.stack(outputs.hidden_states[-4:])
                        mean_embeddings = last_layers.mean(dim=0)
                        cls_embedding = mean_embeddings[:, 0, :]
                        mean_pooled = mean_embeddings.mean(dim=1)
                        combined_embedding = torch.cat([cls_embedding, mean_pooled], dim=1)
                        features['embeddings'].append(combined_embedding.cpu())

        # Combine features - SENTIMENT AND EMBEDDINGS ONLY
        combined_features = {
            'sentiment': torch.cat(features['sentiment']).numpy(),
            'embeddings': torch.cat(features['embeddings']).numpy()
        }

        # Enhanced sentiment scoring
        sentiment_scores = combined_features['sentiment'][:, 2] - combined_features['sentiment'][:, 0]
        sentiment_confidence = np.max(combined_features['sentiment'], axis=1)

        return {
            'sentiment_score': sentiment_scores,
            'sentiment_confidence': sentiment_confidence,
            'sentiment_raw': combined_features['sentiment'],
            'embeddings': combined_features['embeddings']
        }

# --------------------- VAE Training with Curves ---------------------
def train_vae_with_curves(vae, data, device, epochs=30, batch_size=256):
    """Train VAE with comprehensive loss tracking"""
    optimizer = optim.Adam(vae.parameters(), lr=config.vae_lr)
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae.train()
    losses = {'total': [], 'recon': [], 'kl': []}

    for epoch in range(epochs):
        epoch_recon, epoch_kl, epoch_total = 0, 0, 0
        for batch in loader:
            x = batch[0].to(device)
            recon, mu, logvar = vae(x)

            recon_loss = F.mse_loss(recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            kl_weight = min(1.0, (epoch + 1) / (epochs * 0.3))
            loss = recon_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            epoch_total += loss.item()

        losses['recon'].append(epoch_recon/len(dataset))
        losses['kl'].append(epoch_kl/len(dataset))
        losses['total'].append(epoch_total/len(dataset))

        if (epoch + 1) % 10 == 0:
            print(f"VAE Epoch {epoch+1}/{epochs} - Total Loss: {epoch_total/len(dataset):.4f}")

    # Plot VAE training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(losses['total'], label='Total Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Total Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(losses['recon'], label='Reconstruction Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Reconstruction Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(losses['kl'], label='KL Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE KL Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vae_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    vae.eval()
    with torch.no_grad():
        full_data = torch.tensor(data, dtype=torch.float32).to(device)
        mu, _ = vae.encode(full_data)
        return vae, mu.cpu().numpy(), losses

# --------------------- Enhanced Training with Comprehensive Curves ---------------------
def train_with_curves(model, X_train, y_train, X_val, y_val, device, batch_size=64, epochs=100, lr=3e-4):
    """Enhanced training with comprehensive curve tracking"""
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'learning_rates': []
    }

    best_f1 = 0
    patience, wait = 10, 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        all_train_preds, all_train_labels = [], []

        for batch_x, batch_y in train_loader:
            # Mixup augmentation
            if torch.rand(1) < 0.7:
                lam = np.random.beta(0.4, 0.4)
                idx = torch.randperm(batch_x.size(0))
                batch_x = lam * batch_x + (1 - lam) * batch_x[idx]
                target_a, target_b = batch_y, batch_y[idx]
            else:
                lam = 1.0

            optimizer.zero_grad()
            outputs = model(batch_x)

            if lam == 1.0:
                loss = criterion(outputs, batch_y)
            else:
                loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(batch_y.cpu().numpy())

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size*2)

            for val_batch_x, val_batch_y in val_loader:
                val_outputs = model(val_batch_x)
                val_loss_batch = criterion(val_outputs, val_batch_y)
                val_loss += val_loss_batch.item()

                val_preds = val_outputs.argmax(dim=1)
                val_correct += (val_preds == val_batch_y).sum().item()
                val_total += val_batch_y.size(0)

                all_val_preds.extend(val_preds.cpu().numpy())
                all_val_labels.extend(val_batch_y.cpu().numpy())

        # Calculate comprehensive metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        train_precision = precision_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro', zero_division=0)

        # Update history
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(val_loss/len(val_loader) if val_total > 0 else 0)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['learning_rates'].append(scheduler.get_last_lr()[0])

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {history['train_loss'][-1]:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {history['val_loss'][-1]:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  LR: {history['learning_rates'][-1]:.2e}")

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")
            wait = 0
            print(f"  🎯 New best model saved with F1: {best_f1:.4f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"  ⏹️ Early stopping at epoch {epoch+1}")
                break

    return history, best_f1

def plot_comprehensive_training_curves(history):
    """Comprehensive training curves visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1-score curves
    axes[0, 2].plot(history['train_f1'], label='Train F1', linewidth=2)
    axes[0, 2].plot(history['val_f1'], label='Val F1', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1-Score')
    axes[0, 2].set_title('Training and Validation F1-Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Precision curves
    axes[1, 0].plot(history['train_precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Training and Validation Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Recall curves
    axes[1, 1].plot(history['train_recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Training and Validation Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1, 2].plot(history['learning_rates'], color='purple', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comprehensive_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a summary performance plot
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(history['train_acc']) + 1)

    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy', alpha=0.7)
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy', alpha=0.7)
    plt.plot(epochs_range, history['train_f1'], label='Train F1', alpha=0.7)
    plt.plot(epochs_range, history['val_f1'], label='Val F1', alpha=0.7)

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Model Training Summary - Accuracy and F1-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_summary_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    print("COMPREHENSIVE MODEL EVALUATION")
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
    plt.title('Confusion Matrix - Bullying Type Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_multiclass.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('probability_analysis_multiclass.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Confidence distribution
    plt.figure(figsize=(10, 6))
    confidence_scores = np.max(y_probs, axis=1)
    predicted_classes = np.argmax(y_probs, axis=1)

    for i, class_name in enumerate(class_names):
        class_confidences = confidence_scores[predicted_classes == i]
        if len(class_confidences) > 0:
            plt.hist(class_confidences, bins=20, alpha=0.7,
                    label=f'{class_name}', color=f'C{i}')

    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Model Confidence Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_distribution_multiclass.png', dpi=300, bbox_inches='tight')
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

# --------------------- Enhanced Captum Explanations ---------------------
def get_sentiment_interpretation(impact):
    """Interpretation for sentiment features"""
    if impact > 0:
        return "Negative sentiment reinforces bullying detection"
    else:
        return "Positive sentiment counters bullying classification"

def get_latent_interpretation(feature_name, impact):
    """Interpretation for latent features"""
    feature_num = int(feature_name.split('_')[-1])
    
    # Group latent features by potential semantic meaning
    if feature_num < 16:
        interpretation = "Semantic pattern related to aggressive language"
    elif feature_num < 32:
        interpretation = "Contextual understanding feature"
    elif feature_num < 48:
        interpretation = "Linguistic structure pattern"
    else:
        interpretation = "Deep semantic representation"
    
    direction = "reinforces" if impact > 0 else "counters"
    return f"{interpretation} → {direction} bullying classification"

def enhanced_captum_explain(model, X_test_final, y_test, feature_names, class_names, label_encoder, texts=None):
    """Enhanced Captum explanations for multi-class bullying detection"""
    print(f"\nRunning enhanced Captum explanation on {min(config.captum_samples, len(X_test_final))} samples...")

    ig = IntegratedGradients(model)

    for i in range(min(config.captum_samples, len(X_test_final))):
        input_tensor = torch.tensor(X_test_final[i:i+1], dtype=torch.float32).to(config.device)

        # Create intelligent baseline
        baseline = torch.zeros_like(input_tensor)
        # Sentiment baseline is zero (neutral)

        with torch.no_grad():
            outputs = model(input_tensor)
            pred_probs = F.softmax(outputs, dim=1)
            pred_class_idx = pred_probs.argmax().item()
            pred_class = class_names[pred_class_idx]
            confidence = pred_probs[0][pred_class_idx].item()
            true_class = class_names[y_test[i]]

        try:
            # Compute attributions
            attributions, delta = ig.attribute(
                input_tensor,
                baselines=baseline,
                target=pred_class_idx,
                return_convergence_delta=True,
                n_steps=50
            )

            attributions = attributions.squeeze().detach().cpu().numpy()

            # Get top influential features
            top_indices = np.argsort(np.abs(attributions))[::-1][:10]

            print(f"\n📊 Sample {i+1} Explanation:")
            if texts is not None:
                print(f"📝 Text: {texts[i][:100]}...")
            print(f"✅ True Label: {true_class}")
            print(f"🔮 Predicted: {pred_class} (confidence: {confidence:.2f})")
            print(f"📋 Description: {config.bullying_descriptions[pred_class]}")
            print("🔥 Most Influential Features:")

            for rank, idx in enumerate(top_indices, 1):
                if idx >= len(feature_names):
                    continue

                impact = attributions[idx]
                direction = "↑ promotes" if impact > 0 else "↓ suppresses"
                color = "\033[91m" if impact > 0 else "\033[92m"

                feature_name = feature_names[idx]

                if feature_name == "sentiment_score":
                    display_name = "Sentiment Score"
                    interpretation = get_sentiment_interpretation(impact)
                elif feature_name.startswith("latent_"):
                    display_name = f"Latent Feature {feature_name.split('_')[-1]}"
                    interpretation = get_latent_interpretation(feature_name, impact)
                else:
                    display_name = feature_name
                    interpretation = f"Feature pattern {'supports' if impact > 0 else 'counters'} {pred_class} classification"

                print(f"{rank}. {color}{display_name} {direction}\033[0m (impact: {impact:.4f})")
                print(f"   💡 {interpretation}")

            # Create enhanced visualization
            plot_enhanced_attributions_multiclass(attributions, feature_names, top_indices,
                                                true_class, pred_class, confidence,
                                                texts[i] if texts else None, i+1)

        except Exception as e:
            print(f"❌ Error explaining sample {i+1}: {e}")
            continue

def plot_enhanced_attributions_multiclass(attributions, feature_names, top_indices, true_label, pred_label, confidence, text, sample_num):
    """Create enhanced attribution plots for multi-class"""
    plt.figure(figsize=(14, 8))

    # Prepare data
    plot_features = []
    plot_impacts = []
    colors = []

    for idx in reversed(top_indices):
        if idx >= len(feature_names):
            continue
        impact = attributions[idx]

        # Convert feature name to readable format
        feature_name = feature_names[idx]
        if feature_name == "sentiment_score":
            readable_name = "Sentiment Score"
        elif feature_name.startswith("latent_"):
            readable_name = f"Latent Feature {feature_name.split('_')[-1]}"
        else:
            readable_name = feature_name

        plot_features.append(readable_name)
        plot_impacts.append(impact)
        colors.append('red' if impact > 0 else 'green')

    # Create horizontal bar plot
    y_pos = np.arange(len(plot_features))
    bars = plt.barh(y_pos, plot_impacts, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for bar, impact in zip(bars, plot_impacts):
        width = bar.get_width()
        label_x = width if width > 0 else width
        ha = 'left' if width > 0 else 'right'
        plt.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{impact:.3f}', va='center', ha=ha, fontsize=10, fontweight='bold')

    plt.yticks(y_pos, plot_features)
    plt.xlabel('Feature Impact Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Enhanced title
    title = f"Sample {sample_num}\nTrue: {true_label}, Predicted: {pred_label} (Confidence: {confidence:.2f})"
    if text:
        title = f"{title}\nText: {text[:80]}..."
    plt.title(title, fontsize=14, pad=20)

    # Add legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Promotes classification')
    green_patch = mpatches.Patch(color='green', alpha=0.7, label='Suppresses classification')
    plt.legend(handles=[red_patch, green_patch], loc='lower right')

    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'enhanced_captum_sample_{sample_num}_multiclass.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📈 Saved enhanced visualization: enhanced_captum_sample_{sample_num}_multiclass.png")

# --------------------- Cross-Validation ---------------------
def perform_cross_validation(X, y, model_class, device, n_splits=5):
    """Perform stratified k-fold cross-validation"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    fold_metrics = {
        'accuracy': [], 'f1': [], 'precision': [], 'recall': [],
        'train_loss': [], 'val_loss': []
    }

    print(f"\nPerforming {n_splits}-fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Prepare tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

        # Initialize and train model
        model = model_class(X_train.shape[1], len(np.unique(y))).to(device)
        history, best_f1 = train_with_curves(
            model, X_train_t, y_train_t, X_val_t, y_val_t, device,
            epochs=50, lr=config.classifier_lr
        )

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1).cpu().numpy()

        fold_metrics['accuracy'].append(accuracy_score(y_val, preds))
        fold_metrics['f1'].append(f1_score(y_val, preds, average='macro'))
        fold_metrics['precision'].append(precision_score(y_val, preds, average='macro', zero_division=0))
        fold_metrics['recall'].append(recall_score(y_val, preds, average='macro', zero_division=0))

        print(f"Fold {fold + 1} - Accuracy: {fold_metrics['accuracy'][-1]:.4f}, F1: {fold_metrics['f1'][-1]:.4f}")

    # Print CV results
    print(f"\n📊 Cross-Validation Results ({n_splits}-fold):")
    for metric, values in fold_metrics.items():
        if metric in ['accuracy', 'f1', 'precision', 'recall']:
            print(f"{metric.capitalize()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    return fold_metrics

# --------------------- Model Saving and Loading ---------------------
def save_all_artifacts(model, vae, scaler, label_encoder, feature_extractor, config, feature_names):
    """Save all models and artifacts for external evaluation"""
    artifacts = {
        'model_state_dict': model.state_dict(),
        'vae_state_dict': vae.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'config': config,
        'model_architecture': 'EnhancedDNN',
        'feature_names': feature_names,
        'feature_extractor_info': {
            'sentiment_model': config.model_config['sentiment']['model_name'],
            'embedding_model': config.model_config['embedding']['model_name'],
            'fine_tuned_models_used': True
        }
    }

    # Save individual artifacts
    torch.save(model.state_dict(), 'bullying_classifier_model.pt')
    torch.save(vae.state_dict(), 'vae_model.pt')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(artifacts, 'all_artifacts.pkl')

    # Save concatenated features info
    feature_info = {
        'total_features': len(feature_names),
        'sentiment_features': 1,
        'latent_features': 64,
        'feature_names': feature_names,
        'feature_breakdown': {
            'sentiment': ['sentiment_score'],
            'latent': [f'latent_{i}' for i in range(64)]
        }
    }
    joblib.dump(feature_info, 'feature_info.pkl')

    # Save model info
    model_info = {
        'bullying_types': config.clean_bullying_types,
        'bullying_descriptions': config.bullying_descriptions,
        'input_dim': model.input_bn.num_features,
        'num_classes': model.output.out_features,
        'feature_names': feature_names,
        'fine_tuned_models_used': True,
        'total_features': len(feature_names)
    }
    joblib.dump(model_info, 'model_info.pkl')

    print("✅ All artifacts saved successfully!")
    print("📁 Saved files:")
    print("  - bullying_classifier_model.pt (Main classifier)")
    print("  - vae_model.pt (VAE for embeddings)")
    print("  - scaler.pkl (Feature scaler)")
    print("  - label_encoder.pkl (Label encoder)")
    print("  - all_artifacts.pkl (All artifacts combined)")
    print("  - feature_info.pkl (Feature information)")
    print("  - model_info.pkl (Model configuration)")

# --------------------- Enhanced Main Pipeline ---------------------
def main():
    print(f"Using device: {config.device}")
    print("🚀 USING SENTIMENT AND EMBEDDING MODELS ONLY!")

    try:
        # Load and balance data with proper distribution
        train_df, val_df, test_df, label_encoder = load_and_balance_data("1.csv")

        # Initialize feature extractor with SENTIMENT AND EMBEDDING MODELS
        feature_extractor = RobustFeatureExtractor(config)

        # Extract features for each split separately
        print("\nExtracting features for training set...")
        train_features = feature_extractor.extract_features(train_df['cleaned_text'].tolist())

        print("Extracting features for validation set...")
        val_features = feature_extractor.extract_features(val_df['cleaned_text'].tolist())

        print("Extracting features for test set...")
        test_features = feature_extractor.extract_features(test_df['cleaned_text'].tolist())

        # Train VAE on training data only
        print("\nTraining VAE on training data only...")
        vae = ImprovedVAE(input_dim=train_features['embeddings'].shape[1]).to(config.device)
        vae, train_latent, vae_losses = train_vae_with_curves(
            vae, train_features['embeddings'], config.device
        )

        # Transform validation and test using trained VAE
        with torch.no_grad():
            val_embeddings_t = torch.tensor(val_features['embeddings'], dtype=torch.float32).to(config.device)
            test_embeddings_t = torch.tensor(test_features['embeddings'], dtype=torch.float32).to(config.device)

            val_latent, _ = vae.encode(val_embeddings_t)
            test_latent, _ = vae.encode(test_embeddings_t)

            val_latent = val_latent.cpu().numpy()
            test_latent = test_latent.cpu().numpy()

        # Prepare feature matrices - SENTIMENT + LATENT FEATURES ONLY
        def prepare_features(sentiment_score, latent):
            return np.concatenate([sentiment_score.reshape(-1, 1), latent], axis=1)

        X_train = prepare_features(
            train_features['sentiment_score'],
            train_latent
        )
        X_val = prepare_features(
            val_features['sentiment_score'],
            val_latent
        )
        X_test = prepare_features(
            test_features['sentiment_score'],
            test_latent
        )

        y_train = train_df['label_encoded'].values
        y_val = val_df['label_encoded'].values
        y_test = test_df['label_encoded'].values

        print(f"\n📊 Feature Dimensions:")
        print(f"Training features: {X_train.shape}")
        print(f"Validation features: {X_val.shape}")
        print(f"Test features: {X_test.shape}")
        print(f"Total features: {X_train.shape[1]} (1 sentiment + 64 latent = 65)")

        # Standardization on training data only
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train)
        X_val_final = scaler.transform(X_val)
        X_test_final = scaler.transform(X_test)

        # Create feature names for interpretability
        feature_names = ['sentiment_score'] + [f'latent_{i}' for i in range(train_latent.shape[1])]

        # Save concatenated features
        concatenated_features = {
            'X_train': X_train_final,
            'X_val': X_val_final, 
            'X_test': X_test_final,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_names,
            'feature_breakdown': {
                'sentiment_features': 1,
                'latent_features': train_latent.shape[1],
                'total_features': len(feature_names)
            }
        }
        joblib.dump(concatenated_features, 'concatenated_features.pkl')
        print("✅ Saved concatenated features to 'concatenated_features.pkl'")

        # Convert to tensors
        X_train_t = torch.tensor(X_train_final, dtype=torch.float32).to(config.device)
        X_val_t = torch.tensor(X_val_final, dtype=torch.float32).to(config.device)
        X_test_t = torch.tensor(X_test_final, dtype=torch.float32).to(config.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(config.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(config.device)
        y_test_t = torch.tensor(y_test, dtype=torch.long).to(config.device)

        # Perform cross-validation
        cv_results = perform_cross_validation(
            X_train_final, y_train, EnhancedDNN, config.device, n_splits=5
        )

        # Train final model on full training data
        print("\nTraining final model on full training set...")
        final_model = EnhancedDNN(X_train_final.shape[1], len(config.clean_bullying_types)).to(config.device)
        history, best_f1 = train_with_curves(
            final_model, X_train_t, y_train_t, X_val_t, y_val_t, config.device,
            epochs=config.classifier_epochs, lr=config.classifier_lr
        )

        # Plot comprehensive training curves
        plot_comprehensive_training_curves(history)

        # Load best model and evaluate on test set
        final_model.load_state_dict(torch.load("best_model.pt"))

        # Comprehensive evaluation
        class_names = config.clean_bullying_types
        test_results = comprehensive_evaluation(final_model, X_test_t, y_test_t, class_names, label_encoder)

        print(f"\n🎯 FINAL TEST RESULTS (on completely isolated test set):")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"Macro F1: {test_results['macro_f1']:.4f}")
        print(f"Weighted F1: {test_results['weighted_f1']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")

        # Compare with cross-validation results
        print(f"\n📊 CROSS-VALIDATION vs TEST PERFORMANCE:")
        print(f"CV Accuracy: {np.mean(cv_results['accuracy']):.4f} ± {np.std(cv_results['accuracy']):.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"CV F1-Score: {np.mean(cv_results['f1']):.4f} ± {np.std(cv_results['f1']):.4f}")
        print(f"Test F1-Score: {test_results['macro_f1']:.4f}")

        # Enhanced Captum explanations
        print(f"\n🔍 Running enhanced interpretability analysis...")
        enhanced_captum_explain(
            final_model, X_test_final, y_test, feature_names, class_names, label_encoder, test_df['cleaned_text'].tolist()
        )

        # Save all artifacts for external evaluation
        save_all_artifacts(final_model, vae, scaler, label_encoder, feature_extractor, config, feature_names)

        print("\n✅ All models and artifacts saved successfully!")
        print("🎯 FINE-TUNED MODELS USED:")
        print("  - Fine-tuned Sentiment Model: best_sentiment_model.pt")
        print("  - Fine-tuned Embedding Model: best_embedding_model.pt")
        print(f"📊 Using {len(feature_names)} features (1 sentiment + 64 latent)")
        print("\n📈 Generated visualizations:")
        print("  - comprehensive_training_curves.png")
        print("  - training_summary_curves.png")
        print("  - vae_training_curves.png")
        print("  - confusion_matrix_multiclass.png")
        print("  - probability_analysis_multiclass.png")
        print("  - confidence_distribution_multiclass.png")
        print("  - enhanced_captum_sample_*_multiclass.png")

    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()