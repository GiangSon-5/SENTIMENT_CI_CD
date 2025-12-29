import pandas as pd
import yaml
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def train():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Setup MLflow
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    print("🚀 Starting training...")
    
    # Load processed data
    X_train = pd.read_pickle(os.path.join(config['data']['processed_path'], "X_train.pkl"))
    X_test = pd.read_pickle(os.path.join(config['data']['processed_path'], "X_test.pkl"))
    y_train = pd.read_pickle(os.path.join(config['data']['processed_path'], "y_train.pkl"))
    y_test = pd.read_pickle(os.path.join(config['data']['processed_path'], "y_test.pkl"))

    with mlflow.start_run():
        # Params
        C_param = config['train']['C']
        mlflow.log_param("C", C_param)
        mlflow.log_param("model_type", "LogisticRegression")

        # Train
        model = LogisticRegression(C=C_param)
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')

        print(f"📊 Metrics: Accuracy={acc:.4f}, F1={f1:.4f}")

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log Model & Vectorizer
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(config['train']['vectorizer_path'])

        # Save model locally for API
        with open(config['train']['model_path'], "wb") as f:
            pickle.dump(model, f)
        
        print(f"✅ Training finished. Model saved to {config['train']['model_path']}")

if __name__ == "__main__":
    train()