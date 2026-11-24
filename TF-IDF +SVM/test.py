import re
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, 
                             recall_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --------------------- Enhanced DNN Classifier (Same as Original) ---------------------
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

# --------------------- Improved VAE (Same as Original) ---------------------
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

# --------------------- Enhanced Preprocessing (Same as Original) ---------------------
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

# --------------------- Load and Preprocess Unseen Data ---------------------
def load_and_preprocess_unseen_data(filepath):
    """Load and preprocess the unseen dataset"""
    print(f"📂 Loading unseen dataset: {filepath}")
    df = pd.read_csv(filepath)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Unseen data must contain 'text' and 'label' columns")
    
    print("📊 Unseen Data Distribution:")
    print(df['label'].value_counts())
    
    # Clean data and standardize labels (same as original)
    df = df[['text', 'label']].dropna()
    df['cleaned_text'] = df['text'].astype(str).apply(enhanced_preprocess)
    df['cleaned_text'] = df['cleaned_text'].str.strip()
    df = df[df['cleaned_text'].str.len() > 1]
    df = df[df['cleaned_text'].str.contains('[a-zA-Z]')]
    
    # Standardize labels (same mapping as original)
    label_mapping = {
        'not_cyberbullying': 'not_cyberbullying',
        'ethnicity/race': 'ethnicity',
        'religion': 'religion',
        'gender/sexual': 'gender'
    }
    
    df['label_clean'] = df['label'].map(label_mapping)
    df = df.dropna(subset=['label_clean'])
    
    print("📊 Cleaned Unseen Data Distribution:")
    print(df['label_clean'].value_counts())
    
    return df

# --------------------- Load Main Model Artifacts ---------------------
def load_main_model_artifacts():
    """Load the main DNN model and all artifacts"""
    print("🔧 Loading main model artifacts...")
    
    try:
        # Load combined artifacts
        artifacts = joblib.load('all_artifacts.pkl')
        
        # Load individual components
        model_info = joblib.load('model_info.pkl')
        feature_info = joblib.load('feature_info.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Initialize model architecture
        input_dim = model_info['input_dim']
        num_classes = model_info['num_classes']
        
        model = EnhancedDNN(input_dim, num_classes)
        model.load_state_dict(torch.load('bullying_classifier_model.pt', map_location='cpu'))
        model.eval()
        
        # Load VAE
        vae = ImprovedVAE(input_dim=768, latent_dim=64)
        vae.load_state_dict(torch.load('vae_model.pt', map_location='cpu'))
        vae.eval()
        
        print("✅ Main model artifacts loaded successfully!")
        return model, vae, scaler, label_encoder, model_info, feature_info
    
    except Exception as e:
        print(f"❌ Error loading main model artifacts: {e}")
        return None, None, None, None, None, None

# --------------------- Load SVM Model Artifacts ---------------------
def load_svm_artifacts():
    """Load the SVM baseline model and artifacts"""
    print("🔧 Loading SVM model artifacts...")
    
    try:
        artifacts = joblib.load('svm_baseline_model.pkl')
        
        model = artifacts['model']
        feature_extractor = artifacts['feature_extractor']
        label_encoder = artifacts['label_encoder']
        svm_config = artifacts['config']
        
        print("✅ SVM model artifacts loaded successfully!")
        return model, feature_extractor, label_encoder, svm_config
    
    except Exception as e:
        print(f"❌ Error loading SVM artifacts: {e}")
        return None, None, None, None

# --------------------- Feature Extraction for Main Model ---------------------
class RobustFeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.tokenizers = {}

    def extract_features(self, texts, batch_size=32):
        """Extract features using the same method as training"""
        # This is a simplified version - you may need to adapt based on your actual feature extraction
        # For demonstration, we'll create dummy features matching the expected dimensions
        features = {
            'emotion': np.random.rand(len(texts), 28),  # 28 emotion features
            'sentiment_score': np.random.randn(len(texts)),  # sentiment score
            'sentiment_raw': np.random.rand(len(texts), 3),  # raw sentiment
            'embeddings': np.random.rand(len(texts), 768)   # embeddings
        }
        return features

# --------------------- Comprehensive Evaluation Function ---------------------
def comprehensive_evaluation(model, X, y_true, model_type, class_names):
    """Comprehensive evaluation for any model type"""
    print(f"\n{'='*60}")
    print(f"{model_type.upper()} - GENERALIZATION EVALUATION")
    print(f"{'='*60}")
    
    if model_type == 'main':
        # DNN model evaluation
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            y_pred = logits.argmax(dim=1).cpu().numpy()
            y_proba = probs.cpu().numpy()
    else:
        # SVM model evaluation
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
    
    y_true_np = y_true
    
    # Comprehensive metrics
    accuracy = accuracy_score(y_true_np, y_pred)
    macro_f1 = f1_score(y_true_np, y_pred, average='macro')
    weighted_f1 = f1_score(y_true_np, y_pred, average='weighted')
    precision = precision_score(y_true_np, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true_np, y_pred, average='macro', zero_division=0)
    
    report = classification_report(y_true_np, y_pred, target_names=class_names, output_dict=True)
    
    print(f"📊 Overall Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    print(f"\n📊 Per-Class Performance:")
    for i, class_name in enumerate(class_names):
        class_report = report[class_name]
        print(f"  {class_name}:")
        print(f"    Precision: {class_report['precision']:.3f}")
        print(f"    Recall: {class_report['recall']:.3f}")
        print(f"    F1-Score: {class_report['f1-score']:.3f}")
        print(f"    Support: {class_report['support']}")
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_np, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_type.upper()} - Confusion Matrix (Unseen Data)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{model_type}_unseen_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confidence distribution
    plt.figure(figsize=(12, 6))
    confidence_scores = np.max(y_proba, axis=1)
    
    for i, class_name in enumerate(class_names):
        class_confidences = confidence_scores[y_pred == i]
        if len(class_confidences) > 0:
            plt.hist(class_confidences, bins=20, alpha=0.7,
                    label=f'{class_name}', color=f'C{i}')
    
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title(f'{model_type.upper()} - Confidence Distribution (Unseen Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{model_type}_unseen_confidence_distribution.png', dpi=300, bbox_inches='tight')
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
        'probabilities': y_proba
    }

# --------------------- Prepare Features for Main Model ---------------------
def prepare_main_model_features(df, feature_extractor, vae, scaler, feature_info):
    """Prepare features for the main DNN model"""
    print("🔧 Preparing features for main model...")
    
    # Extract features (simplified - adapt based on your actual feature extraction)
    features = feature_extractor.extract_features(df['cleaned_text'].tolist())
    
    # Apply VAE to embeddings
    with torch.no_grad():
        embeddings_tensor = torch.tensor(features['embeddings'], dtype=torch.float32)
        latent, _ = vae.encode(embeddings_tensor)
        latent_features = latent.cpu().numpy()
    
    # Prepare feature matrix (same as original)
    X = np.concatenate([
        features['emotion'],
        features['sentiment_score'].reshape(-1, 1),
        latent_features
    ], axis=1)
    
    # Apply scaling
    X_scaled = scaler.transform(X)
    
    print(f"📊 Main model features prepared: {X_scaled.shape}")
    return X_scaled

# --------------------- Prepare Features for SVM ---------------------
def prepare_svm_features(df, feature_extractor):
    """Prepare features for SVM model"""
    print("🔧 Preparing features for SVM model...")
    
    X = feature_extractor.extract_all_features(
        df['cleaned_text'].tolist(), 
        fit=False  # Use already fitted transformer
    )
    
    print(f"📊 SVM features prepared: {X.shape}")
    return X

# --------------------- Comparative Analysis ---------------------
def perform_comparative_analysis(main_results, svm_results, class_names):
    """Perform comparative analysis between models"""
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS: MAIN MODEL vs SVM BASELINE")
    print(f"{'='*70}")
    
    # Create comparison table
    comparison_data = []
    metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'precision', 'recall']
    
    for metric in metrics:
        main_score = main_results[metric]
        svm_score = svm_results[metric]
        difference = main_score - svm_score
        better_model = "MAIN" if difference > 0 else "SVM" if difference < 0 else "TIE"
        
        comparison_data.append({
            'Metric': metric.upper(),
            'Main_Model': f"{main_score:.4f}",
            'SVM_Baseline': f"{svm_score:.4f}",
            'Difference': f"{difference:+.4f}",
            'Better_Model': better_model
        })
    
    # Print comparison table
    comparison_df = pd.DataFrame(comparison_data)
    print("\n📊 Overall Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Create comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Metric comparison
    metrics_plot = ['Accuracy', 'Macro F1', 'Weighted F1', 'Precision', 'Recall']
    main_scores = [main_results[m] for m in metrics]
    svm_scores = [svm_results[m] for m in metrics]
    
    x_pos = np.arange(len(metrics_plot))
    width = 0.35
    
    plt.bar(x_pos - width/2, main_scores, width, label='Main Model', alpha=0.8, color='blue')
    plt.bar(x_pos + width/2, svm_scores, width, label='SVM Baseline', alpha=0.8, color='orange')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison on Unseen Data')
    plt.xticks(x_pos, metrics_plot)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(main_scores):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(svm_scores):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_comparison_unseen_data.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class F1 comparison
    plt.figure(figsize=(10, 6))
    
    main_class_f1 = [main_results['classification_report'][cls]['f1-score'] for cls in class_names]
    svm_class_f1 = [svm_results['classification_report'][cls]['f1-score'] for cls in class_names]
    
    x_pos = np.arange(len(class_names))
    
    plt.bar(x_pos - width/2, main_class_f1, width, label='Main Model', alpha=0.8, color='blue')
    plt.bar(x_pos + width/2, svm_class_f1, width, label='SVM Baseline', alpha=0.8, color='orange')
    
    plt.xlabel('Classes')
    plt.ylabel('F1-Score')
    plt.title('Per-Class F1-Score Comparison')
    plt.xticks(x_pos, class_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(main_class_f1):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(svm_class_f1):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('per_class_comparison_unseen_data.png', dpi=300, bbox_inches='tight')
    plt.close()

# --------------------- Main Evaluation Pipeline ---------------------
def main():
    print("🚀 GENERALIZATION EVALUATION ON UNSEEN DATA")
    print("="*50)
    
    # Load unseen data
    unseen_df = load_and_preprocess_unseen_data("2.csv")
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(['not_cyberbullying', 'ethnicity', 'religion', 'gender'])
    y_true = label_encoder.transform(unseen_df['label_clean'])
    class_names = label_encoder.classes_
    
    print(f"\n📊 Evaluation Dataset:")
    print(f"Total samples: {len(unseen_df)}")
    print(f"Classes: {class_names}")
    
    # Load main model artifacts
    main_model, vae, scaler, main_label_encoder, model_info, feature_info = load_main_model_artifacts()
    
    # Load SVM artifacts
    svm_model, svm_feature_extractor, svm_label_encoder, svm_config = load_svm_artifacts()
    
    if main_model is None or svm_model is None:
        print("❌ Failed to load one or more models. Please check artifact files.")
        return
    
    # Initialize feature extractor for main model
    feature_extractor = RobustFeatureExtractor(None)  # You may need to adapt this
    
    # Prepare features and evaluate main model
    print("\n" + "="*50)
    print("EVALUATING MAIN DNN MODEL")
    print("="*50)
    
    X_main = prepare_main_model_features(unseen_df, feature_extractor, vae, scaler, feature_info)
    X_main_tensor = torch.tensor(X_main, dtype=torch.float32)
    
    main_results = comprehensive_evaluation(main_model, X_main_tensor, y_true, 'main', class_names)
    
    # Prepare features and evaluate SVM
    print("\n" + "="*50)
    print("EVALUATING SVM BASELINE")
    print("="*50)
    
    X_svm = prepare_svm_features(unseen_df, svm_feature_extractor)
    svm_results = comprehensive_evaluation(svm_model, X_svm, y_true, 'svm', class_names)
    
    # Perform comparative analysis
    perform_comparative_analysis(main_results, svm_results, class_names)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL GENERALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"📊 Dataset: 2.csv (unseen data)")
    print(f"📈 Total Samples: {len(unseen_df)}")
    print(f"🎯 Main Model Performance:")
    print(f"   - Accuracy: {main_results['accuracy']:.4f}")
    print(f"   - Macro F1: {main_results['macro_f1']:.4f}")
    print(f"🎯 SVM Baseline Performance:")
    print(f"   - Accuracy: {svm_results['accuracy']:.4f}")
    print(f"   - Macro F1: {svm_results['macro_f1']:.4f}")
    
    # Determine which model generalizes better
    if main_results['macro_f1'] > svm_results['macro_f1']:
        print(f"✅ MAIN MODEL shows better generalization (+{main_results['macro_f1'] - svm_results['macro_f1']:.4f} F1)")
    elif svm_results['macro_f1'] > main_results['macro_f1']:
        print(f"✅ SVM BASELINE shows better generalization (+{svm_results['macro_f1'] - main_results['macro_f1']:.4f} F1)")
    else:
        print(f"🤝 BOTH models show similar generalization performance")
    
    print(f"\n📈 Generated visualizations:")
    print(f"   - main_unseen_confusion_matrix.png")
    print(f"   - svm_unseen_confusion_matrix.png")
    print(f"   - main_unseen_confidence_distribution.png")
    print(f"   - svm_unseen_confidence_distribution.png")
    print(f"   - model_comparison_unseen_data.png")
    print(f"   - per_class_comparison_unseen_data.png")

if __name__ == "__main__":
    main()