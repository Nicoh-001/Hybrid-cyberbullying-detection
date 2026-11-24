import re
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer
import xgboost as xgb
from scipy import sparse
import joblib
import json

warnings.filterwarnings('ignore')

# --------------------- Configuration ---------------------
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = 42
        self.bert_model_name = 'bert-base-uncased'
        self.max_len = 128
        self.batch_size = 256
        self.tfidf_max_features = 300
        self.tfidf_ngram_range = (1, 2)
        self.bert_reduced_dim = 200

config = Config()

# --------------------- Preprocessing ---------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', 'USER', text)
    text = re.sub(r'#\w+', 'HASHTAG', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --------------------- Data Loading ---------------------
def load_data(filepath):
    print("Loading data...")
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=50000):
        chunk = chunk[['text', 'label']].dropna()
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    
    print(f"Original dataset size: {len(df)}")
    
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    df = df[df['cleaned_text'].str.len() > 1]
    
    label_mapping = {
        'not_cyberbullying': 'not_cyberbullying',
        'ethnicity': 'ethnicity', 
        'religion': 'religion' ,
        'gender': 'gender' ,
        'age': 'age' ,
        'other_cyberbullying': 'other_cyberbullying'
    }
    df['label_clean'] = df['label'].map(label_mapping)
    df = df.dropna(subset=['label_clean'])
    
    print("Class distribution:")
    print(df['label_clean'].value_counts())
    
    return df

# --------------------- BERT Feature Extractor ---------------------
class BERTFeatureExtractor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
        self.model = BertModel.from_pretrained(config.bert_model_name).to(config.device)
        self.model.eval()
    
    def extract_features_batch(self, texts):
        features = []
        
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(text) for text in texts]
        
        for i in range(0, len(texts), config.batch_size):
            batch = texts[i:i+config.batch_size]
            
            inputs = self.tokenizer(
                batch, return_tensors='pt', padding=True, 
                truncation=True, max_length=config.max_len
            ).to(config.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                last_hidden_state = outputs.last_hidden_state
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                
                reduced_embeddings = mean_embeddings[:, :config.bert_reduced_dim]
                features.append(reduced_embeddings.cpu().numpy())
            
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.vstack(features)

# --------------------- TF-IDF Feature Extractor ---------------------
class TFIDFFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=config.tfidf_ngram_range,
            dtype=np.float32
        )
    
    def fit_transform(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(text) for text in texts]
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(text) for text in texts]
        return self.vectorizer.transform(texts)

# --------------------- Model Saver ---------------------
def save_model_artifacts(tfidf_extractor, bert_extractor, xgb_model, label_encoder, config):
    """Save all model artifacts for future inference"""
    
    joblib.dump(tfidf_extractor.vectorizer, 'tfidf_vectorizer.pkl')
    bert_extractor.tokenizer.save_pretrained('./bert_tokenizer')
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    config_dict = {
        'bert_model_name': config.bert_model_name,
        'max_len': config.max_len,
        'batch_size': config.batch_size,
        'tfidf_max_features': config.tfidf_max_features,
        'tfidf_ngram_range': config.tfidf_ngram_range,
        'bert_reduced_dim': config.bert_reduced_dim,
        'device': str(config.device)
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("✅ Model artifacts saved successfully!")

# --------------------- Inference Class ---------------------
class CyberbullyingClassifier:
    def __init__(self, model_dir='.'):
        self.model_dir = model_dir
        self.load_models()
    
    def load_models(self):
        """Load all saved model artifacts"""
        print("Loading model artifacts...")
        
        self.tfidf_vectorizer = joblib.load(f'{self.model_dir}/tfidf_vectorizer.pkl')
        self.bert_tokenizer = BertTokenizer.from_pretrained(f'{self.model_dir}/bert_tokenizer')
        self.bert_model = BertModel.from_pretrained(config.bert_model_name).to(config.device)
        self.bert_model.eval()
        self.xgb_model = joblib.load(f'{self.model_dir}/xgboost_model.pkl')
        self.label_encoder = joblib.load(f'{self.model_dir}/label_encoder.pkl')
        
        with open(f'{self.model_dir}/model_config.json', 'r') as f:
            self.config = json.load(f)
        
        print("✅ All models loaded successfully!")
    
    def preprocess_text(self, text):
        """Preprocess text for inference"""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', 'USER', text)
        text = re.sub(r'#\w+', 'HASHTAG', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', 'NUM', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_bert_features(self, texts):
        """Extract BERT features for inference"""
        features = []
        texts = [str(text) for text in texts]
        
        for i in range(0, len(texts), self.config['batch_size']):
            batch = texts[i:i+self.config['batch_size']]
            
            inputs = self.bert_tokenizer(
                batch, return_tensors='pt', padding=True, 
                truncation=True, max_length=self.config['max_len']
            ).to(config.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                attention_mask = inputs['attention_mask']
                last_hidden_state = outputs.last_hidden_state
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                
                reduced_embeddings = mean_embeddings[:, :self.config['bert_reduced_dim']]
                features.append(reduced_embeddings.cpu().numpy())
        
        return np.vstack(features)
    
    def predict(self, texts):
        """Make predictions on new texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
        bert_features = self.extract_bert_features(processed_texts)
        
        combined_features = sparse.hstack([tfidf_features, sparse.csr_matrix(bert_features)]).tocsr()
        
        predictions = self.xgb_model.predict(combined_features)
        probabilities = self.xgb_model.predict_proba(combined_features)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return {
            'predictions': predicted_labels,
            'probabilities': probabilities,
            'class_names': self.label_encoder.classes_.tolist()
        }

# --------------------- Main Pipeline ---------------------
def main():
    print("🚀 TF-IDF + BERT + XGBoost Cyberbullying Classification")
    print("📊 Reduced features: 300 TF-IDF + 200 BERT = 500 total")
    
    # Load data
    df = load_data("1.csv")
    print(f"Final dataset size: {len(df)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label_clean'])
    
    # Split data - ONLY TRAIN/TEST
    print("Splitting data (Train/Test only)...")
    texts = df['cleaned_text'].astype(str).tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2,
        stratify=y, random_state=config.random_state
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Extract features
    print("\nExtracting TF-IDF features...")
    tfidf_extractor = TFIDFFeatureExtractor()
    X_train_tfidf = tfidf_extractor.fit_transform(X_train)
    X_test_tfidf = tfidf_extractor.transform(X_test)
    
    print("Extracting BERT features...")
    bert_extractor = BERTFeatureExtractor()
    X_train_bert = bert_extractor.extract_features_batch(X_train)
    X_test_bert = bert_extractor.extract_features_batch(X_test)
    
    # Combine features
    X_train_combined = sparse.hstack([X_train_tfidf, sparse.csr_matrix(X_train_bert)]).tocsr()
    X_test_combined = sparse.hstack([X_test_tfidf, sparse.csr_matrix(X_test_bert)]).tocsr()
    
    print(f"Final feature dimensions:")
    print(f"TF-IDF: {X_train_tfidf.shape[1]}")
    print(f"BERT: {X_train_bert.shape[1]}")
    print(f"Total: {X_train_combined.shape[1]}")
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.random_state,
        eval_metric='mlogloss',
        tree_method='hist'
    )
    
    xgb_model.fit(X_train_combined, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = xgb_model.predict(X_test_combined)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Print results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    
    # Per-class metrics
    class_names = label_encoder.classes_
    print(f"\nPer-class F1 Scores:")
    f1_scores = f1_score(y_test, y_pred, average=None)
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {f1_scores[i]:.4f}")
    
    # Save model artifacts
    print("\nSaving model artifacts...")
    save_model_artifacts(tfidf_extractor, bert_extractor, xgb_model, label_encoder, config)
    
    print("✅ Training completed! Model is ready for unseen data evaluation.")

if __name__ == "__main__":
    main()