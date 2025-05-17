import requests
import os

def download_file(url, destination):
    print(f"Downloading {destination}...")
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download {destination}: HTTP {response.status_code}")

    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type or response.content.startswith(b'<!DOCTYPE html>'):
        raise Exception(f"Received HTML instead of a binary file for {destination}")

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {destination} ({os.path.getsize(destination)} bytes)")

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

files = {
    "historical_features_A.pkl": "https://huggingface.co/datasets/MiguelMendesDs/retail-pricing-models/resolve/main/historical_features_A.pkl",
    "historical_features_B.pkl": "https://huggingface.co/datasets/MiguelMendesDs/retail-pricing-models/resolve/main/historical_features_B.pkl",
    "price_pipeline_A.pkl": "https://huggingface.co/datasets/MiguelMendesDs/retail-pricing-models/resolve/main/price_pipeline_A.pkl",
    "price_pipeline_B.pkl": "https://huggingface.co/datasets/MiguelMendesDs/retail-pricing-models/resolve/main/price_pipeline_B.pkl"
}

# Download all files
for filename, url in files.items():
    try:
        download_file(url, f"./models/{filename}")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
