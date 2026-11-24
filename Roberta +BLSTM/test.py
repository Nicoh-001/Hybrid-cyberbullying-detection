import re

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer


# --------------------- Configuration ---------------------
class Config:
    MAX_LEN = 128
    BATCH_SIZE = 32
    MODEL_NAME = 'roberta-base'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------- Preprocessing ---------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# --------------------- Dataset Class ---------------------
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
        encoding = self.tokenizer.encode_plus(
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
            'label': torch.tensor(self.labels[idx], dtype=torch.long) if self.labels is not None else torch.tensor(-1, dtype=torch.long)
        }

# --------------------- Model Architecture ---------------------
class RoBERTaBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super(RoBERTaBiLSTM, self).__init__()
        self.roberta = RobertaModel.from_pretrained(Config.MODEL_NAME)
        self.lstm = nn.LSTM(
            input_size=self.roberta.config.hidden_size,
            hidden_size=256,  # Should match your trained model's hidden_dim
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3  # Should match your trained model's dropout
        )
        self.dropout = nn.Dropout(0.3)  # Should match your trained model's dropout
        self.classifier = nn.Linear(256 * 2, num_classes)  # Should match your trained model's hidden_dim
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = roberta_output.last_hidden_state
        
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output[:, -1, :])
        
        logits = self.classifier(lstm_output)
        return logits

# --------------------- Test Function ---------------------
def test_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def load_model_and_tokenizer(model_path, tokenizer_path, num_classes):
    model = RoBERTaBiLSTM(num_classes).to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()
    
    tokenizer = joblib.load(tokenizer_path)
    
    return model, tokenizer

def main():
    # Load test data
    test_df = pd.read_csv('test_data.csv')
    test_df['text'] = test_df['headline'].apply(preprocess_text)
    test_df = test_df[test_df['text'].str.len() > 10]
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist() if 'label' in test_df.columns else None
    
    # Load model and tokenizer
    num_classes = len(set(test_labels)) if test_labels is not None else 2  # Adjust if you know the number of classes
    model, tokenizer = load_model_and_tokenizer(
        'roberta_bilstm_best.pt',
        'tokenizer.joblib',
        num_classes
    )
    
    # Create test dataset and dataloader
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, Config.MAX_LEN)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    # Run predictions
    preds, true_labels = test_model(model, test_loader)
    
    # Save predictions if labels not available
    if test_labels is None:
        test_df['predicted_label'] = preds
        test_df.to_csv('test_predictions.csv', index=False)
        print("Predictions saved to test_predictions.csv")
    else:
        # Calculate metrics
        accuracy = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average='macro')
        
        print("\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, preds))
        
        # Save results with predictions
        test_df['predicted_label'] = preds
        test_df.to_csv('test_results_with_predictions.csv', index=False)
        print("\nResults with predictions saved to test_results_with_predictions.csv")

if __name__ == "__main__":
    main()