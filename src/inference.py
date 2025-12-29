import pickle
import yaml
import re

# Load artifacts once (Global scope for caching)
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open(config['train']['model_path'], "rb") as f:
    model = pickle.load(f)

with open(config['train']['vectorizer_path'], "rb") as f:
    vectorizer = pickle.load(f)

def predict_sentiment(text: str):
    # Preprocess text (phải giống lúc train)
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Transform & Predict
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()

    label = "Positive" if prediction == 1 else "Negative"
    # Mapping dataset tùy chỉnh: 0=Negative, 1=Positive (Twitter dataset gốc thường là 0 và 4, ở đây giả sử 0 và 1)
    
    return {"label": label, "confidence": float(proba)}