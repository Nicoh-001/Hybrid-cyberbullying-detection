import re
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def enhanced_preprocess(text):
    """More nuanced text preprocessing preserving important linguistic features"""
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
    text = str(text).lower().strip()
    for pattern, repl in patterns:
        text = pattern.sub(repl, text)
    return text.strip()

def load_and_analyze_data(filepath):
    """Load data with thorough analysis"""
    df = pd.read_csv(filepath)
    
    # Validate structure
    if 'headline' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'headline' and 'label' columns")
    
    # Preprocess
    df['cleaned_text'] = df['headline'].apply(enhanced_preprocess)
    df['label'] = df['label'].astype(int)
    
    # Analyze dataset
    print("\n=== Dataset Analysis ===")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts(normalize=True)}")
    
    # Check text lengths
    text_lengths = df['cleaned_text'].str.split().str.len()
    print(f"\nText length statistics (in words):")
    print(text_lengths.describe())
    
    # Show examples
    print("\nSample positive examples:")
    print(df[df['label'] == 1]['cleaned_text'].head(3).tolist())
    print("\nSample negative examples:")
    print(df[df['label'] == 0]['cleaned_text'].head(3).tolist())
    
    return df

def bert_tokenize(texts, tokenizer, max_length=64):
    """More efficient tokenization with dynamic padding"""
    return tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

def prepare_dataloaders(train_df, val_df, tokenizer, batch_size=32):
    """Prepare dataloaders with proper tensor conversion"""
    # Tokenize texts
    train_encodings = bert_tokenize(train_df['cleaned_text'].tolist(), tokenizer)
    val_encodings = bert_tokenize(val_df['cleaned_text'].tolist(), tokenizer)

    # Create TensorDataset properly
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        torch.tensor(train_df['label'].values, dtype=torch.long)
    )
    
    val_dataset = TensorDataset(
        val_encodings['input_ids'],
        val_encodings['attention_mask'],
        torch.tensor(val_df['label'].values, dtype=torch.long)
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size*2
    )
    
    return train_loader, val_loader

def evaluate(model, dataloader, device):
    """Comprehensive evaluation with accuracy"""
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits
            
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).flatten().cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(inputs['labels'].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\nAdditional Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(true_labels, predictions, digits=4))
    
    return avg_loss, f1, accuracy

def train_model(model, train_loader, val_loader, device, epochs=2, lr=5e-6):
    """More conservative training regimen"""
    model.to(device)
    
    # Smaller learning rate and fewer epochs
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(train_loader) * epochs
    
    # Linear warmup over 10% of training
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    best_f1 = 0
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            
            model.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        # Validation
        val_loss, val_f1, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model

def main():
    # Load and analyze data
    df = load_and_analyze_data("1.csv")  # 99,990 samples
    
    # Simple train-val-test split (80-10-10)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    
    # Create dataloaders
    train_loader, val_loader = prepare_dataloaders(train_df, val_df, tokenizer, batch_size=32)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train model
    model = train_model(model, train_loader, val_loader, device, epochs=3)
    
    # Prepare test loader
    test_encodings = bert_tokenize(test_df['cleaned_text'].tolist(), tokenizer)
    test_dataset = TensorDataset(
        test_encodings['input_ids'],
        test_encodings['attention_mask'],
        torch.tensor(test_df['label'].values, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Final evaluation
    test_loss, test_f1, test_acc = evaluate(model, test_loader, device)
    
    print(f"\nFinal Test Scores:")
    print(f"F1: {test_f1:.4f}")
    print(f"Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()