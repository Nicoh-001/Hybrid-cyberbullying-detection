import re
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
class Config:
    MAX_LENGTH = 64
    BATCH_SIZE = 32
    MODEL_PATH = 'best_model.pt'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def enhanced_preprocess(text):
    """Text preprocessing function matching the training setup"""
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

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load model architecture
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    
    return model, tokenizer

def prepare_test_data(test_df, tokenizer):
    """Prepare test data in the same format as training"""
    # Preprocess text
    test_df['cleaned_text'] = test_df['headline'].apply(enhanced_preprocess)
    
    # Tokenize
    encodings = tokenizer(
        test_df['cleaned_text'].tolist(),
        add_special_tokens=True,
        max_length=Config.MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Create dataset
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(test_df['label'].values, dtype=torch.long) if 'label' in test_df.columns 
        else torch.zeros(len(test_df), dtype=torch.long)  # Dummy labels if none exist
    )
    
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE)

def evaluate_model(model, dataloader):
    """Evaluate model performance"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch[0].to(Config.DEVICE),
                'attention_mask': batch[1].to(Config.DEVICE),
                'labels': batch[2].to(Config.DEVICE)
            }
            
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(inputs['labels'].cpu().numpy())
    
    return predictions, true_labels

def plot_confusion_matrix(true_labels, predictions):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def save_predictions(test_df, predictions):
    """Save predictions to CSV"""
    test_df['prediction'] = predictions
    test_df.to_csv('test_predictions.csv', index=False)
    print("\nSample predictions:")
    print(test_df[['headline', 'prediction']].head())

def main():
    # Load test data
    try:
        test_df = pd.read_csv('test_data.csv')
        print(f"Loaded test data with {len(test_df)} samples")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer()
        print("Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare test data
    test_loader = prepare_test_data(test_df, tokenizer)

    # Run evaluation
    predictions, true_labels = evaluate_model(model, test_loader)

    # Handle results based on whether we have true labels
    if 'label' in test_df.columns:
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='macro')
        
        print("\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        # Plot confusion matrix
        plot_confusion_matrix(true_labels, predictions)
        print("Confusion matrix saved to confusion_matrix.png")
        
        # Save predictions with true labels
        save_predictions(test_df, predictions)
    else:
        # Just save predictions
        print("\nNo labels found in test data - saving predictions only")
        save_predictions(test_df, predictions)

if __name__ == "__main__":
    main()