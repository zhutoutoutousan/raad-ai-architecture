#!/usr/bin/env python3
"""
Simple Hugging Face Upload Script
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file

def upload_to_huggingface():
    """Upload the model to Hugging Face Hub."""
    
    # Configuration
    REPO_NAME = "anonymous/architectural-style-classifier"  # Change this to your username
    TOKEN = os.getenv("HF_TOKEN")  # Set this environment variable
    
    if not TOKEN:
        print("❌ Please set HF_TOKEN environment variable")
        print("   export HF_TOKEN=your_token_here")
        return False
    
    print(f"🚀 Uploading to Hugging Face Hub: {REPO_NAME}")
    
    # Initialize API
    api = HfApi(token=TOKEN)
    
    # Create repository
    try:
        create_repo(REPO_NAME, token=TOKEN, exist_ok=True)
        print(f"✅ Repository created/verified: {REPO_NAME}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return False
    
    # Upload key files
    files_to_upload = [
        ("checkpoints/best_model/efficientnet_b0-epoch=04-val_acc=0.997.ckpt", "pytorch_model.bin"),
        ("README.md", "README.md"),
        ("LICENSE", "LICENSE"),
        ("requirements.txt", "requirements.txt"),
        ("setup.py", "setup.py"),
        ("MANIFEST.in", "MANIFEST.in"),
    ]
    
    # Upload directories
    dirs_to_upload = [
        "src",
        "paper", 
        "results",
        "checkpoints/best_model"
    ]
    
    try:
        # Upload individual files
        for src_path, dest_path in files_to_upload:
            if Path(src_path).exists():
                upload_file(
                    path_or_fileobj=src_path,
                    path_in_repo=dest_path,
                    repo_id=REPO_NAME,
                    token=TOKEN
                )
                print(f"✅ Uploaded: {dest_path}")
            else:
                print(f"⚠️  File not found: {src_path}")
        
        # Upload directories
        for dir_path in dirs_to_upload:
            if Path(dir_path).exists():
                for file_path in Path(dir_path).rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(Path(dir_path))
                        repo_path = f"{dir_path}/{relative_path}"
                        upload_file(
                            path_or_fileobj=str(file_path),
                            path_in_repo=str(repo_path),
                            repo_id=REPO_NAME,
                            token=TOKEN
                        )
                        print(f"✅ Uploaded: {repo_path}")
            else:
                print(f"⚠️  Directory not found: {dir_path}")
        
        print(f"\n🎉 Successfully uploaded to: https://huggingface.co/{REPO_NAME}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading files: {e}")
        return False

if __name__ == "__main__":
    print("🏛️ Architectural Style Classifier - Hugging Face Upload")
    print("=" * 60)
    
    if upload_to_huggingface():
        print("✅ Upload completed successfully!")
    else:
        print("❌ Upload failed")
