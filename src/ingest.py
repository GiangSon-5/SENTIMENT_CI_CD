import pandas as pd
import os
import yaml

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def ingest_data():
    raw_path = config['data']['raw_path']
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    print("🚀 Creating dummy dataset...")
    data = {
        "text": [
            "I love this product, it is amazing!",
            "Terrible service, very disappointed.",
            "Absolutely fantastic experience.",
            "I hate waiting so long.",
            "It is okay, nothing special.",
            "Worst purchase ever!",
            "Highly recommended!",
            "Not bad, but could be better."
        ] * 50, 
        "target": [1, 0, 1, 0, 0, 0, 1, 0] * 50 # 1: Positive, 0: Negative
    }
    
    df = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter4000.csv", encoding='latin1')
    # Chúng ta lấy 1 subset thực tế từ internet cho khách quan hơn
    # Nếu link die, fallback về dummy data
    
    try:
        df = df[['twitts', 'sentiment']].rename(columns={'twitts': 'text', 'sentiment': 'target'})
        df = df.sample(1000) # Lấy 1000 dòng
        print("✅ Downloaded sample data from GitHub.")
    except:
        df = pd.DataFrame(data)
        print("⚠️ Download failed. Using generated dummy data.")

    df.to_csv(raw_path, index=False)
    print(f"✅ Data saved to {raw_path}")

if __name__ == "__main__":
    ingest_data()