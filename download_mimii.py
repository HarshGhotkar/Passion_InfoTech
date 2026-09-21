import os
import requests
import zipfile
import sys

# Zenodo Record 3384388 for MIMII dataset
ZENODO_BASE_URL = "https://zenodo.org/record/3384388/files/"
DATASETS = ["6_dB_slider.zip"]
DOWNLOAD_DIR = r"d:\Passion_InfoTech"

def download_file(url, filepath):
    print(f"Downloading {url} to {filepath}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    with open(filepath, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
            downloaded += len(data)
            if total_size > 0:
                percent = int(50 * downloaded / total_size)
                sys.stdout.write(f"\r[{'=' * percent}{' ' * (50-percent)}] {downloaded / (1024*1024):.2f} MB")
                sys.stdout.flush()
    print("\nDownload complete.")

def extract_zip(filepath, extract_to):
    print(f"Extracting {filepath} to {extract_to}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

if __name__ == "__main__":
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    print("=== MIMII Dataset Downloader ===")
    print("Warning: These files are quite large (several GBs each).")
    
    for filename in DATASETS:
        url = ZENODO_BASE_URL + filename + "?download=1"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        extract_path = os.path.join(DOWNLOAD_DIR, filename.replace('.zip', ''))
        
        if not os.path.exists(filepath):
            try:
                download_file(url, filepath)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                continue
        else:
            print(f"File {filename} already exists, skipping download.")
            
        if not os.path.exists(extract_path):
            try:
                extract_zip(filepath, DOWNLOAD_DIR)
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
        else:
            print(f"Directory {extract_path} already exists, skipping extraction.")
            
    print("All datasets processed.")
