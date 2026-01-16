import os
import sys
from pathlib import Path

def diag():
    print("=== DragGAN Environment Diagnostic ===")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Python Executable: {sys.executable}")
    
    # Check checkpoints directory
    cache_dir = './checkpoints'
    print(f"\nChecking directory: {cache_dir}")
    if os.path.exists(cache_dir):
        print(f"✅ Directory exists.")
        files = os.listdir(cache_dir)
        pkl_files = [f for f in files if f.endswith('.pkl')]
        print(f"Total files: {len(files)}")
        print(f"PKL files found: {len(pkl_files)}")
        for f in pkl_files:
            print(f"  - {f}")
    else:
        print(f"❌ Directory does not exist.")
        
    # Check root directory
    print(f"\nChecking root directory for .pkl files:")
    root_pkls = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if root_pkls:
        print(f"✅ Found {len(root_pkls)} .pkl files in root.")
        for f in root_pkls:
            print(f"  - {f}")
    else:
        print("ℹ️ No .pkl files in root.")

    # Check script existence
    if os.path.exists('scripts/download_model.py'):
        print("\n✅ scripts/download_model.py found.")
    else:
        print("\n❌ scripts/download_model.py NOT found.")

if __name__ == "__main__":
    diag()
