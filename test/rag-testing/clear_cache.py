import os
import sys
import shutil
from huggingface_hub import list_repo_files, get_full_repo_path, delete_repo_tags

# Define the models we suspect caused the large download/corruption
# These are the paths the large model files were saved under
TARGET_MODEL_CACHE_DIRS = [
    "models--microsoft--phi-2",
    "models--meta-llama--Llama-3-8B-Instruct" 
]

def delete_corrupted_cache():
    """Manually deletes local cache directories for suspected large models."""
    print("--- Starting Hugging Face Cache Cleanup ---")
    
    # Determine the default cache directory path
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    if not os.path.exists(cache_dir):
        print(f"FATAL: Hugging Face cache directory not found at {cache_dir}.")
        print("The cache may be in a non-standard location. You may need to delete the cache folder manually.")
        return

    print(f"Scanning cache directory: {cache_dir}")
    
    deleted_count = 0
    
    for target_dir in TARGET_MODEL_CACHE_DIRS:
        model_cache_path = os.path.join(cache_dir, target_dir)
        
        if os.path.exists(model_cache_path):
            print(f"Found corrupted cache directory for: {target_dir}")
            
            try:
                # Recursively delete the directory and its contents
                shutil.rmtree(model_cache_path)
                print(f"✅ Successfully deleted corrupted cache for {target_dir}.")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete directory {model_cache_path}: {e}")
                print("   (Permission issues are common. Please delete this folder manually if this error persists.)")
        else:
            print(f"Cache for {target_dir} not found. Proceeding.")

    if deleted_count > 0:
        print("\nCleanup complete. Corrupted files have been removed.")
    else:
        print("\nNo suspected corrupted caches were found.")

if __name__ == "__main__":
    delete_corrupted_cache()
    print("\n--- Next Step: Rerun RAG Test Script ---")