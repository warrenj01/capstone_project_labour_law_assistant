import os
import random
import json

# --- Configuration ---
# Input directory containing the small, chunked Cabinet Decisions
CABINET_CHUNKS_DIR = "../documents_processed/cabinet_chunks_for_filtering"
# Output file where the labeled dictionary will be saved
OUTPUT_LABELS_FILE = "cabinet_training_labels.py"

# Define the split for the training set
TARGET_POSITIVE_SAMPLES = 25 # High-relevance, keyword-scored chunks
TARGET_NEGATIVE_SAMPLES = 25 # Low-relevance, random chunks
TOTAL_SAMPLES = TARGET_POSITIVE_SAMPLES + TARGET_NEGATIVE_SAMPLES

# List of high-value, domain-specific keywords for ranking
LABOUR_KEYWORDS = [
    'wage', 'salary', 'remuneration', 'employment', 'worker', 'employee', 
    'termination', 'severance', 'prgf', 'retirement', 'redundancy', 'industrial relations' ,'income tax', 'self-employed'
]

def score_all_chunks():
    global CABINET_CHUNKS_DIR, LABOUR_KEYWORDS,OUTPUT_LABELS_FILE,TARGET_POSITIVE_SAMPLES,TARGET_NEGATIVE_SAMPLES,TOTAL_SAMPLES
    """Scores all chunks by keyword relevance and separates them into high/low relevance lists."""
    
    if not os.path.exists(CABINET_CHUNKS_DIR):
        print(f"Error: Input directory not found: {CABINET_CHUNKS_DIR}")
        return [], []
        
    scored_chunks = []
    
    for filename in os.listdir(CABINET_CHUNKS_DIR):
        if filename.endswith('.txt'):
            filepath = os.path.join(CABINET_CHUNKS_DIR, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
            except:
                continue

            # Calculate Score: Count keyword occurrences
            score = sum(content.count(kw) for kw in LABOUR_KEYWORDS)
            
            scored_chunks.append({
                "filename": filename, 
                "content": content,
                "score": score
            })
    
    # Sort by score: highest relevance first
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    
    # Split into high and low relevance groups
    # High: Top N chunks that had keywords
    high_relevance = [c for c in scored_chunks if c['score'] > 0][:TARGET_POSITIVE_SAMPLES]
    # Low: All chunks that had low/zero scores (potential irrelevant pool)
    low_relevance_pool = [c for c in scored_chunks if c['score'] == 0]
    
    return high_relevance, low_relevance_pool

def score_all_chunks():
    """Scores all chunks by keyword relevance and separates them into high/low relevance lists."""
    
    if not os.path.exists(CABINET_CHUNKS_DIR):
        print(f"Error: Input directory not found: {CABINET_CHUNKS_DIR}")
        return [], []
        
    scored_chunks = []
    
    for filename in os.listdir(CABINET_CHUNKS_DIR):
        if filename.endswith('.txt'):
            filepath = os.path.join(CABINET_CHUNKS_DIR, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
            except:
                continue

            # Calculate Score: Count keyword occurrences
            score = sum(content.count(kw) for kw in LABOUR_KEYWORDS)
            
            scored_chunks.append({
                "filename": filename, 
                "content": content,
                "score": score
            })
    
    # Sort by score: highest relevance first
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    
    # Split into high and low relevance groups
    # High: Top N chunks that had keywords
    high_relevance = [c for c in scored_chunks if c['score'] > 0][:TARGET_POSITIVE_SAMPLES]
    # Low: All chunks that had low/zero scores (potential irrelevant pool)
    low_relevance_pool = [c for c in scored_chunks if c['score'] == 0]
    
    return high_relevance, low_relevance_pool

def run_labeling_session():
    """Runs the labeling session, combining high-score and random low-score chunks."""
    
    high_relevance_samples, low_relevance_pool = score_all_chunks()

    if not high_relevance_samples:
        print("FATAL: Could not find any chunks matching Labour Keywords. Check keywords or data.")
        return

    # 1. Select the final samples for labeling
    # We take the top keyword-scored chunks as our positive candidates
    final_samples = high_relevance_samples
    
    # We randomly sample from the low-relevance pool for our negative candidates
    # This ensures a good variety of noise (irrelevant) samples
    num_to_sample_from_low = min(TARGET_NEGATIVE_SAMPLES, len(low_relevance_pool))
    random_low_samples = random.sample(low_relevance_pool, num_to_sample_from_low)
    
    final_samples.extend(random_low_samples)
    random.shuffle(final_samples) # Shuffle the list so you don't label all '1's followed by all '0's

    labeled_data = {}
    
    print("\n" + "="*80)
    print(f"STARTING MANUAL LABELING SESSION (Total: {len(final_samples)} balanced samples)")
    print("Goal: Create a balanced set (Labour vs. Noise) for ML training.")
    print("="*80)
    
    for i, item in enumerate(final_samples):
        filename = item['filename']
        content = item['content']
        score = item.get('score', 0)
        
        print(f"\n[{i+1}/{len(final_samples)}] File: {filename} (Keyword Score: {score})")
        print("-" * 80)
        print(content[:700] + "..." if len(content) > 700 else content)
        print("-" * 80)
        
        # Get user input for label
        while True:
            label_input = input("Label (1=Relevant/Labour, 0=Irrelevant/Noise, s=Skip): ").lower().strip()
            
            if label_input == '1':
                labeled_data[filename] = 1
                break
            elif label_input == '0':
                labeled_data[filename] = 0
                break
            elif label_input == 's':
                break
            else:
                print("Invalid input. Please enter 1, 0, or s.")

    # 2. Save the results as a Python dictionary file
    
    with open(OUTPUT_LABELS_FILE, 'w', encoding='utf-8') as f:
        f.write("# This file was generated by manual_labeling_tool_balanced.py\n")
        f.write(f"# Total Labeled Chunks: {len(labeled_data)}\n")
        f.write("CABINET_TRAINING_LABELS = {\n")
        for filename, label in labeled_data.items():
            f.write(f"    '{filename}': {label},\n")
        f.write("}\n")

    print("\n" + "="*80)
    print("LABELING SESSION COMPLETE")
    print(f"Total labeled items saved for ML Training: {len(labeled_data)}")
    print(f"Results saved to: {os.path.abspath(OUTPUT_LABELS_FILE)}")
    print("="*80)

if __name__ == "__main__":
    run_labeling_session()