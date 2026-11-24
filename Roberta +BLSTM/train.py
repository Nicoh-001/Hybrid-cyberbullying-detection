import re
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

# --------------------- Configuration for 100K Dataset ---------------------
class Config:
    MAX_LEN = 128
    BATCH_SIZE = 32  # Increased for larger dataset
    EPOCHS = 3      # Can train longer with more data
    LEARNING_RATE = 2e-5
    HIDDEN_DIM = 256
    DROPOUT = 0.3
    MODEL_NAME = 'roberta-base'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    TEST_SIZE = 0.2

# Set random seeds
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

# --------------------- Efficient Preprocessing ---------------------
def preprocess_text(text):
    """Efficient text preprocessing for large dataset"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+', 'USER', text)
    text = re.sub(r'#\w+', 'HASHTAG', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# --------------------- Optimized Dataset Class ---------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
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
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --------------------- Model Architecture ---------------------
class RoBERTaBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super(RoBERTaBiLSTM, self).__init__()
        self.roberta = RobertaModel.from_pretrained(Config.MODEL_NAME)
        self.lstm = nn.LSTM(
            input_size=self.roberta.config.hidden_size,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=Config.DROPOUT
        )
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.classifier = nn.Linear(Config.HIDDEN_DIM * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = roberta_output.last_hidden_state
        
        lstm_output, (hidden, cell) = self.lstm(sequence_output)
        
        # Use the last hidden state from both directions
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        output = self.dropout(hidden_combined)
        logits = self.classifier(output)
        return logits

# --------------------- Optimized Training Functions ---------------------
def train_epoch(model, dataloader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(Config.DEVICE)
        attention_mask = batch['attention_mask'].to(Config.DEVICE)
        labels = batch['label'].to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, accuracy, macro_f1

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'labels': all_labels
    }

# --------------------- Data Loading for 100K Dataset ---------------------
def load_large_dataset(filepath):
    """Load dataset without preprocessing to prevent data leakage"""
    print("Loading large dataset...")
    
    # Load data
    try:
        df = pd.read_csv(filepath)
    except:
        # If file is too large, read in chunks
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=50000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    
    print(f"Dataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Use appropriate text column
    text_column = 'text' if 'text' in df.columns else 'headline'
    
    # Get raw texts and labels - NO PREPROCESSING HERE
    raw_texts = df[text_column].astype(str).tolist()
    labels = df['label'].tolist()
    
    # Filter out very short texts based on raw length
    mask = [len(text.strip()) > 10 for text in raw_texts]
    raw_texts = [text for text, keep in zip(raw_texts, mask) if keep]
    labels = [label for label, keep in zip(labels, mask) if keep]
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print("\nDataset Distribution:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = sum(encoded_labels == i)
        print(f"  {class_name}: {count} samples ({count/len(encoded_labels)*100:.2f}%)")
    
    return raw_texts, encoded_labels, label_encoder

def get_class_weights(labels):
    """Compute class weights for imbalanced dataset"""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float).to(Config.DEVICE)

# --------------------- Main Training Function for 100K Dataset ---------------------
def train_roberta_bilstm_large():
    print("🚀 Starting Roberta+BiLSTM Training on 100K Dataset")
    print(f"Using device: {Config.DEVICE}")
    
    # Load raw data (no preprocessing yet)
    raw_texts, labels, label_encoder = load_large_dataset('1.csv')
    
    print(f"\nTotal samples: {len(raw_texts):,}")
    
    # Split data FIRST to prevent data leakage
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        raw_texts, labels, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.SEED,
        stratify=labels
    )
    
    print(f"\nData Split:")
    print(f"Training samples: {len(train_texts):,}")
    print(f"Test samples: {len(test_texts):,}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Preprocess texts SEPARATELY after splitting
    print("\nPreprocessing training texts...")
    train_texts = [preprocess_text(text) for text in tqdm(train_texts, desc="Training texts")]
    
    print("Preprocessing test texts...")
    test_texts = [preprocess_text(text) for text in tqdm(test_texts, desc="Test texts")]
    
    # Check for any overlap between train and test texts
    train_set = set(train_texts)
    test_set = set(test_texts)
    overlap = train_set.intersection(test_set)
    print(f"\nText overlap between train and test: {len(overlap)} samples")
    if overlap:
        print("Warning: Some texts appear in both training and test sets!")
    
    # Initialize tokenizer and datasets
    print("\nInitializing tokenizer and datasets...")
    tokenizer = RobertaTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, Config.MAX_LEN)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, Config.MAX_LEN)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, num_workers=2)
    
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    num_classes = len(label_encoder.classes_)
    model = RoBERTaBiLSTM(num_classes).to(Config.DEVICE)
    
    # Use class weights to handle imbalance (only from training data)
    class_weights = get_class_weights(train_labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'test_loss': [], 'test_acc': [], 'test_f1': []
    }
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    
    print(f"\nStarting training for {Config.EPOCHS} epochs...")
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        print(f"{'='*60}")
        
        # Training
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, scheduler
        )
        
        # Evaluation
        test_results = evaluate_model(model, test_loader, criterion)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['test_loss'].append(test_results['loss'])
        history['test_acc'].append(test_results['accuracy'])
        history['test_f1'].append(test_results['macro_f1'])
        
        # Print results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Test  - Loss: {test_results['loss']:.4f}, Acc: {test_results['accuracy']:.4f}, F1: {test_results['macro_f1']:.4f}")
        print(f"Test Precision: {test_results['precision']:.4f}, Recall: {test_results['recall']:.4f}")
        
        # Save best model
        if test_results['macro_f1'] > best_f1:
            best_f1 = test_results['macro_f1']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_roberta_bilstm_large.pt')
            joblib.dump(label_encoder, 'label_encoder_large.pkl')
            print(f"✅ New best model saved! (F1: {best_f1:.4f})")
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load('best_roberta_bilstm_large.pt'))
    final_results = evaluate_model(model, test_loader, criterion)
    
    # Print final results
    print(f"\n{'='*70}")
    print("🎯 FINAL RESULTS - Roberta+BiLSTM on 100K Dataset")
    print(f"{'='*70}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Test Accuracy: {final_results['accuracy']:.4f}")
    print(f"Test Macro F1: {final_results['macro_f1']:.4f}")
    print(f"Test Weighted F1: {final_results['weighted_f1']:.4f}")
    print(f"Test Precision: {final_results['precision']:.4f}")
    print(f"Test Recall: {final_results['recall']:.4f}")
    
    # Classification report
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(
        final_results['labels'], 
        final_results['predictions'],
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # Save results
    results = {
        'model_name': 'Roberta+BiLSTM_Large',
        'dataset_size': len(raw_texts),
        'final_metrics': {
            'accuracy': final_results['accuracy'],
            'macro_f1': final_results['macro_f1'],
            'weighted_f1': final_results['weighted_f1'],
            'precision': final_results['precision'],
            'recall': final_results['recall']
        },
        'best_epoch': best_epoch,
        'predictions': final_results['predictions'],
        'true_labels': final_results['labels'],
        'class_names': label_encoder.classes_.tolist(),
        'data_leakage_prevention': 'Preprocessing applied after train-test split'
    }
    
    joblib.dump(results, 'roberta_bilstm_large_results.pkl')
    
    print(f"\n✅ Training completed on {len(raw_texts):,} samples!")
    print(f"💾 Saved files:")
    print(f"  - best_roberta_bilstm_large.pt (model weights)")
    print(f"  - label_encoder_large.pkl (label encoder)")
    print(f"  - roberta_bilstm_large_results.pkl (results)")
    
    return results

# Run the training
if __name__ == "__main__":
    results = train_roberta_bilstm_large()