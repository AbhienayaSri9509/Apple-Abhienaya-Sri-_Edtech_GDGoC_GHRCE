import os
import requests
import shutil

# Target directory
TARGET_DIR = os.path.join(os.path.dirname(__file__), "smplx")
os.makedirs(TARGET_DIR, exist_ok=True)

# Sources
# Using Hugging Face mirror from camenduru/SMPLer-X
BASE_URL = "https://huggingface.co/camenduru/SMPLer-X/resolve/main"
MODELS = [
    "SMPLX_NEUTRAL.npz",
    "SMPLX_MALE.npz",
    "SMPLX_FEMALE.npz"
]

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        if os.path.exists(dest_path):
            print(f"  File exists, backing up to {dest_path}.bak")
            shutil.copy2(dest_path, dest_path + ".bak")
            
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("  Download complete.")
        return True
    except Exception as e:
        print(f"  Failed to download: {e}")
        return False

def main():
    print("Starting SMPL-X model download...")
    
    success_count = 0
    for model_name in MODELS:
        url = f"{BASE_URL}/{model_name}"
        dest_path = os.path.join(TARGET_DIR, model_name)
        
        if download_file(url, dest_path):
            success_count += 1
            
            # Verify file size is reasonable (not an error page)
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")
            
            if size_mb < 0.1:
                print("  WARNING: File seems too small, might be invalid.")
    
    if success_count == 0:
        print("\nNo models were downloaded successfully.")
        exit(1)
        
    print(f"\nSuccessfully downloaded {success_count}/{len(MODELS)} models.")
    print("You can debugging tools to verify structure if needed.")

if __name__ == "__main__":
    main()
