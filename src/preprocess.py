import pandas as pd
import numpy as np
import yaml
import os
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("🚀 Starting preprocessing...")
    df = pd.read_csv(config['data']['raw_path'])
    
    # 1. Simple cleaning
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text) # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special chars
        return text

    df['text'] = df['text'].apply(clean_text)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['target'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # 3. Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=config['train']['max_features'])
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Save Artifacts
    os.makedirs(config['data']['processed_path'], exist_ok=True)
    os.makedirs(os.path.dirname(config['train']['vectorizer_path']), exist_ok=True)

    # Save vectorizer (QUAN TRỌNG: để dùng cho inference sau này)
    with open(config['train']['vectorizer_path'], "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save processed data
    pd.to_pickle(X_train_vec, os.path.join(config['data']['processed_path'], "X_train.pkl"))
    pd.to_pickle(X_test_vec, os.path.join(config['data']['processed_path'], "X_test.pkl"))
    pd.to_pickle(y_train, os.path.join(config['data']['processed_path'], "y_train.pkl"))
    pd.to_pickle(y_test, os.path.join(config['data']['processed_path'], "y_test.pkl"))

    print("✅ Preprocessing done. Vectorizer and data saved.")

if __name__ == "__main__":
    preprocess()