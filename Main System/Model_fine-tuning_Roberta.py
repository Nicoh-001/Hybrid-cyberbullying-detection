import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings('ignore')

# Enhanced preprocessing function
def enhanced_preprocess(text):
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

class CyberbullyingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_and_prepare_data(filepath):
    """Load and prepare data for fine-tuning"""
    df = pd.read_csv(filepath)

    # Clean and preprocess
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

    return df

def prepare_sentiment_labels(df):
    """Convert bullying labels to sentiment labels (0-2 range)"""
    sentiment_mapping = {
        'not_cyberbullying': 1,  # neutral
        'ethnicity': 0,          # negative
        'religion': 0,           # negative
        'gender': 0              # negative
    }
    df['sentiment_idx'] = df['label_clean'].map(sentiment_mapping)
    return df

def clear_gpu_memory():
    """Clear GPU memory between models"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ==================== EMOTION FEATURE EXTRACTOR (28D) ====================
class EmotionFeatureExtractor(nn.Module):
    def __init__(self, base_model, feature_dim=768, output_dim=28):
        super(EmotionFeatureExtractor, self).__init__()
        self.base_model = base_model
        # Project to 28 emotion dimensions
        self.emotion_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:, 0, :]
        emotion_features = self.emotion_head(features)
        return emotion_features

def train_emotion_extractor(model, train_loader, val_loader, device, num_epochs=3):
    """Train emotion feature extractor using contrastive learning"""
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    criterion = nn.CosineEmbeddingLoss()
    accumulation_steps = 2

    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f'Emotion - Epoch {epoch+1}/{num_epochs} [Train]')

        for i, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            features = model(input_ids=input_ids, attention_mask=attention_mask)
            
            batch_size = features.size(0)
            if batch_size >= 2:
                loss = 0
                pair_count = 0
                
                for j in range(batch_size):
                    for k in range(j + 1, batch_size):
                        # FIX: Include both positive and negative pairs
                        target = torch.tensor([1.0 if labels[j] == labels[k] else -1.0]).to(device)
                        pair_loss = criterion(
                            features[j].unsqueeze(0), 
                            features[k].unsqueeze(0), 
                            target
                        )
                        loss += pair_loss
                        pair_count += 1
                
                if pair_count > 0:
                    loss = loss / pair_count
                    loss = loss / accumulation_steps
                    loss.backward()
                    total_loss += loss.item() * accumulation_steps
                    total_batches += 1

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if pair_count > 0:
                train_pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / total_batches if total_batches > 0 else float('inf')

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                features = model(input_ids=input_ids, attention_mask=attention_mask)
                
                batch_size = features.size(0)
                if batch_size >= 2:
                    batch_loss = 0
                    pair_count = 0
                    
                    for j in range(batch_size):
                        for k in range(j + 1, batch_size):
                            # FIX: Include both positive and negative pairs
                            target = torch.tensor([1.0 if labels[j] == labels[k] else -1.0]).to(device)
                            pair_loss = criterion(
                                features[j].unsqueeze(0), 
                                features[k].unsqueeze(0), 
                                target
                            )
                            batch_loss += pair_loss.item()
                            pair_count += 1
                    
                    if pair_count > 0:
                        val_loss += batch_loss / pair_count
                        val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')

        print(f'Emotion - Epoch {epoch+1}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss and val_batches > 0:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_emotion_model.pt')
            print(f'  ✅ Saved best emotion model')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  ⏹️ Early stopping for emotion model')
                break

    return model

# ==================== SENTIMENT FEATURE EXTRACTOR (3D) ====================
class SentimentFeatureExtractor(nn.Module):
    def __init__(self, base_model, feature_dim=768, output_dim=3):
        super(SentimentFeatureExtractor, self).__init__()
        self.base_model = base_model
        self.sentiment_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:, 0, :]
        sentiment_features = self.sentiment_head(features)
        return sentiment_features

def train_sentiment_extractor(model, train_loader, val_loader, device, num_epochs=3):
    """Train sentiment feature extractor using contrastive learning"""
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    criterion = nn.CosineEmbeddingLoss()
    accumulation_steps = 2

    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f'Sentiment - Epoch {epoch+1}/{num_epochs} [Train]')

        for i, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            features = model(input_ids=input_ids, attention_mask=attention_mask)
            
            batch_size = features.size(0)
            if batch_size >= 2:
                loss = 0
                pair_count = 0
                
                for j in range(batch_size):
                    for k in range(j + 1, batch_size):
                        # FIX: Include both positive and negative pairs
                        target = torch.tensor([1.0 if labels[j] == labels[k] else -1.0]).to(device)
                        pair_loss = criterion(
                            features[j].unsqueeze(0), 
                            features[k].unsqueeze(0), 
                            target
                        )
                        loss += pair_loss
                        pair_count += 1
                
                if pair_count > 0:
                    loss = loss / pair_count
                    loss = loss / accumulation_steps
                    loss.backward()
                    total_loss += loss.item() * accumulation_steps
                    total_batches += 1

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if pair_count > 0:
                train_pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / total_batches if total_batches > 0 else float('inf')

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                features = model(input_ids=input_ids, attention_mask=attention_mask)
                
                batch_size = features.size(0)
                if batch_size >= 2:
                    batch_loss = 0
                    pair_count = 0
                    
                    for j in range(batch_size):
                        for k in range(j + 1, batch_size):
                            # FIX: Include both positive and negative pairs
                            target = torch.tensor([1.0 if labels[j] == labels[k] else -1.0]).to(device)
                            pair_loss = criterion(
                                features[j].unsqueeze(0), 
                                features[k].unsqueeze(0), 
                                target
                            )
                            batch_loss += pair_loss.item()
                            pair_count += 1
                    
                    if pair_count > 0:
                        val_loss += batch_loss / pair_count
                        val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')

        print(f'Sentiment - Epoch {epoch+1}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss and val_batches > 0:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_sentiment_model.pt')
            print(f'  ✅ Saved best sentiment model')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  ⏹️ Early stopping for sentiment model')
                break

    return model

# ==================== EMBEDDING FEATURE EXTRACTOR (768D) - FIXED ====================
class EmbeddingFeatureExtractor(nn.Module):
    def __init__(self, base_model, feature_dim=768, output_dim=768):
        super(EmbeddingFeatureExtractor, self).__init__()
        self.base_model = base_model
        # Simple projection to 768D - using only CLS token
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use only CLS token for simplicity and compatibility
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        final_embedding = self.embedding_head(cls_embedding)
        return final_embedding

def train_embedding_extractor(model, train_loader, val_loader, device, num_epochs=3):
    """Train embedding feature extractor using contrastive learning"""
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    criterion = nn.CosineEmbeddingLoss()
    accumulation_steps = 2

    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f'Embedding - Epoch {epoch+1}/{num_epochs} [Train]')

        for i, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            features = model(input_ids=input_ids, attention_mask=attention_mask)
            
            batch_size = features.size(0)
            if batch_size >= 2:
                loss = 0
                pair_count = 0
                
                for j in range(batch_size):
                    for k in range(j + 1, batch_size):
                        # FIX: Include both positive and negative pairs
                        target = torch.tensor([1.0 if labels[j] == labels[k] else -1.0]).to(device)
                        pair_loss = criterion(
                            features[j].unsqueeze(0), 
                            features[k].unsqueeze(0), 
                            target
                        )
                        loss += pair_loss
                        pair_count += 1
                
                if pair_count > 0:
                    loss = loss / pair_count
                    loss = loss / accumulation_steps
                    loss.backward()
                    total_loss += loss.item() * accumulation_steps
                    total_batches += 1

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if pair_count > 0:
                train_pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / total_batches if total_batches > 0 else float('inf')

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                features = model(input_ids=input_ids, attention_mask=attention_mask)
                
                batch_size = features.size(0)
                if batch_size >= 2:
                    batch_loss = 0
                    pair_count = 0
                    
                    for j in range(batch_size):
                        for k in range(j + 1, batch_size):
                            # FIX: Include both positive and negative pairs
                            target = torch.tensor([1.0 if labels[j] == labels[k] else -1.0]).to(device)
                            pair_loss = criterion(
                                features[j].unsqueeze(0), 
                                features[k].unsqueeze(0), 
                                target
                            )
                            batch_loss += pair_loss.item()
                            pair_count += 1
                    
                    if pair_count > 0:
                        val_loss += batch_loss / pair_count
                        val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')

        print(f'Embedding - Epoch {epoch+1}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss and val_batches > 0:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_embedding_model.pt')
            print(f'  ✅ Saved best embedding model')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  ⏹️ Early stopping for embedding model')
                break

    return model

# ==================== FINE-TUNING FUNCTIONS ====================
def fine_tune_emotion_model(df, device):
    """Fine-tune the emotion model to output 28D emotion features"""
    print("🔮 Fine-tuning Emotion Model (28D)...")

    model_name = 'SamLowe/roberta-base-go_emotions'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    base_model = RobertaModel.from_pretrained(model_name).to(device)

    model = EmotionFeatureExtractor(base_model, output_dim=28).to(device)

    label_mapping = {
        'not_cyberbullying': 0,
        'ethnicity': 1,
        'religion': 2,
        'gender': 3
    }
    df['label_idx'] = df['label_clean'].map(label_mapping)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label_idx']
    )

    train_dataset = CyberbullyingDataset(
        train_df['cleaned_text'].tolist(),
        train_df['label_idx'].tolist(),
        tokenizer
    )
    val_dataset = CyberbullyingDataset(
        val_df['cleaned_text'].tolist(),
        val_df['label_idx'].tolist(),
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = train_emotion_extractor(model, train_loader, val_loader, device, num_epochs=3)

    return model, tokenizer

def fine_tune_sentiment_model(df, device):
    """Fine-tune the sentiment model to output 3D sentiment features"""
    print("😊 Fine-tuning Sentiment Model (3D)...")

    df_sentiment = prepare_sentiment_labels(df)

    model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    base_model = RobertaModel.from_pretrained(model_name).to(device)

    model = SentimentFeatureExtractor(base_model, output_dim=3).to(device)

    train_df, val_df = train_test_split(
        df_sentiment,
        test_size=0.2,
        random_state=42,
        stratify=df_sentiment['sentiment_idx']
    )

    train_dataset = CyberbullyingDataset(
        train_df['cleaned_text'].tolist(),
        train_df['sentiment_idx'].tolist(),
        tokenizer
    )
    val_dataset = CyberbullyingDataset(
        val_df['cleaned_text'].tolist(),
        val_df['sentiment_idx'].tolist(),
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = train_sentiment_extractor(model, train_loader, val_loader, device, num_epochs=3)

    return model, tokenizer

def fine_tune_embedding_model(df, device):
    """Fine-tune the base model to output 768D embedding features"""
    print("📚 Fine-tuning Embedding Model (768D)...")

    label_mapping = {
        'not_cyberbullying': 0,
        'ethnicity': 1,
        'religion': 2,
        'gender': 3
    }
    df['label_idx'] = df['label_clean'].map(label_mapping)

    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    base_model = RobertaModel.from_pretrained(model_name).to(device)

    model = EmbeddingFeatureExtractor(base_model, output_dim=768).to(device)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label_idx']
    )

    train_dataset = CyberbullyingDataset(
        train_df['cleaned_text'].tolist(),
        train_df['label_idx'].tolist(),
        tokenizer
    )
    val_dataset = CyberbullyingDataset(
        val_df['cleaned_text'].tolist(),
        val_df['label_idx'].tolist(),
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = train_embedding_extractor(model, train_loader, val_loader, device, num_epochs=3)

    return model, tokenizer

# ==================== FEATURE EXTRACTION FUNCTIONS ====================
def extract_emotion_features(model, tokenizer, texts, device, max_len=128):
    """Extract 28D emotion features from texts"""
    model.eval()
    features = []
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            feature = model(input_ids=input_ids, attention_mask=attention_mask)
            features.append(feature.cpu().numpy())
    
    return np.vstack(features)

def extract_sentiment_features(model, tokenizer, texts, device, max_len=128):
    """Extract 3D sentiment features from texts"""
    model.eval()
    features = []
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            feature = model(input_ids=input_ids, attention_mask=attention_mask)
            features.append(feature.cpu().numpy())
    
    return np.vstack(features)

def extract_embedding_features(model, tokenizer, texts, device, max_len=128):
    """Extract 768D embedding features from texts"""
    model.eval()
    features = []
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            feature = model(input_ids=input_ids, attention_mask=attention_mask)
            features.append(feature.cpu().numpy())
    
    return np.vstack(features)

def main():
    """Main function to fine-tune all three models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("📊 Loading dataset...")
    df = load_and_prepare_data("1.csv")

    print(f"Dataset size: {len(df)}")
    print("Label distribution:")
    print(df['label_clean'].value_counts())

    print("\n" + "="*60)
    print("STARTING MODEL FINE-TUNING")
    print("="*60)

    try:
        # Fine-tune emotion model (28D)
        emotion_model, emotion_tokenizer = fine_tune_emotion_model(df, device)
        
        sample_texts = df['cleaned_text'].head(3).tolist()
        emotion_features = extract_emotion_features(emotion_model, emotion_tokenizer, sample_texts, device)
        print(f"✅ Emotion features shape: {emotion_features.shape} (28D raw features)")
        
        del emotion_model, emotion_tokenizer
        clear_gpu_memory()

        # Fine-tune sentiment model (3D)
        sentiment_model, sentiment_tokenizer = fine_tune_sentiment_model(df, device)
        
        sentiment_features = extract_sentiment_features(sentiment_model, sentiment_tokenizer, sample_texts, device)
        print(f"✅ Sentiment features shape: {sentiment_features.shape} (3D raw features)")
        
        del sentiment_model, sentiment_tokenizer
        clear_gpu_memory()

        # Fine-tune embedding model (768D)
        embedding_model, embedding_tokenizer = fine_tune_embedding_model(df, device)
        
        embedding_features = extract_embedding_features(embedding_model, embedding_tokenizer, sample_texts, device)
        print(f"✅ Embedding features shape: {embedding_features.shape} (768D contextual embeddings)")
        
        del embedding_model, embedding_tokenizer
        clear_gpu_memory()

        print(f"\n🎯 Combined feature dimensions:")
        print(f"   Emotion: {emotion_features.shape[1]}D")
        print(f"   Sentiment: {sentiment_features.shape[1]}D") 
        print(f"   Embedding: {embedding_features.shape[1]}D")
        print(f"   Total: {emotion_features.shape[1] + sentiment_features.shape[1] + embedding_features.shape[1]}D")

    except Exception as e:
        print(f"❌ Error during model fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return

    print("\n" + "="*60)
    print("MODEL FINE-TUNING COMPLETED!")
    print("="*60)
    print("✅ Saved models:")
    print("   - best_emotion_model.pt")
    print("   - best_sentiment_model.pt") 
    print("   - best_embedding_model.pt")

    clear_gpu_memory()

if __name__ == "__main__":
    main()