import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer

warnings.filterwarnings('ignore')

# --------------------- Configuration ---------------------
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = 42

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

# --------------------- Model Loader ---------------------
class CyberbullyingClassifier:
    def __init__(self, model_dir='.'):
        self.model_dir = model_dir
        self.load_models()
    
    def load_models(self):
        """Load all saved model artifacts"""
        print("Loading model artifacts...")
        
        # Load TF-IDF
        self.tfidf_vectorizer = joblib.load(f'{self.model_dir}/tfidf_vectorizer.pkl')
        
        # Load BERT components
        self.bert_tokenizer = BertTokenizer.from_pretrained(f'{self.model_dir}/bert_tokenizer')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(config.device)
        self.bert_model.eval()
        
        # Load XGBoost model
        self.xgb_model = joblib.load(f'{self.model_dir}/xgboost_model.pkl')
        
        # Load label encoder
        self.label_encoder = joblib.load(f'{self.model_dir}/label_encoder.pkl')
        
        # Load config
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
        
        batch_size = self.config['batch_size']
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
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
    
    def evaluate_dataset(self, texts, true_labels):
        """Evaluate on a new dataset"""
        print("Making predictions...")
        results = self.predict(texts)
        predictions = self.label_encoder.transform(results['predictions'])
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        weighted_f1 = f1_score(true_labels, predictions, average='weighted')
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        
        # Detailed classification report
        class_report = classification_report(true_labels, predictions, 
                                           target_names=self.label_encoder.classes_, 
                                           output_dict=True)
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'precision': precision,
            'recall': recall,
            'classification_report': class_report,
            'predictions': results['predictions'],
            'probabilities': results['probabilities'],
            'true_labels': true_labels
        }

# --------------------- External Dataset Evaluation ---------------------
def evaluate_external_dataset(csv_file_path):
    """Evaluate the saved model on an external dataset"""
    print(f"🚀 Evaluating model on external dataset: {csv_file_path}")
    
    # Load the external dataset
    print("Loading external dataset...")
    try:
        df_external = pd.read_csv(csv_file_path)
        print(f"External dataset size: {len(df_external)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Check required columns
    if 'text' not in df_external.columns or 'label' not in df_external.columns:
        print("❌ Dataset must contain 'text' and 'label' columns")
        return
    
    # Clean and preprocess
    print("Preprocessing external dataset...")
    df_external = df_external[['text', 'label']].dropna()
    df_external['cleaned_text'] = df_external['text'].apply(preprocess_text)
    df_external = df_external[df_external['cleaned_text'].str.len() > 1]
    
    # Standardize labels (same mapping as training)
    label_mapping = {
         'not_cyberbullying': 'not_cyberbullying',
        'ethnicity': 'ethnicity', 
        'religion': 'religion' ,
        'gender': 'gender' ,
        'age': 'age' ,
        'other_cyberbullying': 'other_cyberbullying'
    }
    df_external['label_clean'] = df_external['label'].map(label_mapping)
    df_external = df_external.dropna(subset=['label_clean'])
    
    print("External dataset class distribution:")
    print(df_external['label_clean'].value_counts())
    
    # Load the trained model
    print("\nLoading trained model...")
    try:
        classifier = CyberbullyingClassifier()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure all model artifacts are in the current directory:")
        print("  - tfidf_vectorizer.pkl")
        print("  - xgboost_model.pkl")
        print("  - label_encoder.pkl")
        print("  - model_config.json")
        print("  - bert_tokenizer/ directory")
        return
    
    # Prepare data for evaluation
    texts = df_external['cleaned_text'].astype(str).tolist()
    true_labels_clean = df_external['label_clean'].values
    
    # Encode true labels using the loaded label encoder
    try:
        true_labels_encoded = classifier.label_encoder.transform(true_labels_clean)
    except ValueError as e:
        print(f"❌ Label encoding error: {e}")
        print("This might be due to new labels in the external dataset.")
        print("Available labels in model:", classifier.label_encoder.classes_)
        print("Labels in external dataset:", np.unique(true_labels_clean))
        return
    
    # Evaluate
    print("\nEvaluating on external dataset...")
    results = classifier.evaluate_dataset(texts, true_labels_encoded)
    
    # Print comprehensive results
    print("\n" + "="*70)
    print("EXTERNAL DATASET EVALUATION RESULTS")
    print("="*70)
    print(f"Dataset: {csv_file_path}")
    print(f"Samples: {len(texts)}")
    print("\n📊 Overall Metrics:")
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Macro F1:    {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    
    # Per-class detailed results
    print(f"\n📈 Per-Class Performance:")
    class_report = results['classification_report']
    for class_name in classifier.label_encoder.classes_:
        if class_name in class_report:
            metrics = class_report[class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1-score']:.4f}")
            print(f"    Support:   {metrics['support']}")
    
    # Macro and weighted averages
    print(f"\n📋 Averages:")
    print(f"  Macro Avg:")
    print(f"    Precision: {class_report['macro avg']['precision']:.4f}")
    print(f"    Recall:    {class_report['macro avg']['recall']:.4f}")
    print(f"    F1-Score:  {class_report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted Avg:")
    print(f"    Precision: {class_report['weighted avg']['precision']:.4f}")
    print(f"    Recall:    {class_report['weighted avg']['recall']:.4f}")
    print(f"    F1-Score:  {class_report['weighted avg']['f1-score']:.4f}")
    
    # Save predictions to file for further analysis
    output_df = df_external.copy()
    output_df['prediction'] = results['predictions']
    output_df['prediction_confidence'] = np.max(results['probabilities'], axis=1)
    
    # Add probability for each class
    for i, class_name in enumerate(classifier.label_encoder.classes_):
        output_df[f'prob_{class_name}'] = results['probabilities'][:, i]
    
    output_file = 'external_dataset_predictions.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\n💾 Predictions saved to: {output_file}")
    
    return results

# --------------------- Main Evaluation Function ---------------------
def main():
    # Evaluate on external dataset
    results = evaluate_external_dataset('2.csv')
    
    if results:
        print("\n✅ External dataset evaluation completed successfully!")
    else:
        print("\n❌ External dataset evaluation failed!")

if __name__ == "__main__":
    main()