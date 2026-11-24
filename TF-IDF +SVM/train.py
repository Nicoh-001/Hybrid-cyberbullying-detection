import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# Simple preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load data
df = pd.read_csv("1.csv")
df = df[['text', 'label']].dropna()

# Preprocess
df['cleaned_text'] = df['text'].apply(preprocess_text)
df = df[df['cleaned_text'].str.len() > 10]

# Encode labels
label_mapping = {
    'not_cyberbullying': 'not_cyberbullying',
    'ethnicity/race': 'ethnicity', 
    'religion': 'religion',
    'gender/sexual': 'gender'
}
df['label_clean'] = df['label'].map(label_mapping)
df = df.dropna(subset=['label_clean'])

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label_clean'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label_encoded'], 
    test_size=0.2, random_state=42, stratify=df['label_encoded']
)

# Create features - keep it small to avoid memory issues
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = svm.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
joblib.dump({'model': svm, 'vectorizer': vectorizer, 'label_encoder': le}, 'svm_baseline.pkl')