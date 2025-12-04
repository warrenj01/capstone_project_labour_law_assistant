import pandas as pd
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_score, recall_score, f1_score

# --- CONFIGURATION (UPDATED TO MATCH YOUR FOLDER STRUCTURE) ---

# Directories containing the already documents - no filter
WRA_CASES_DIRS = [
    "../documents_processed/workers_rights_act_2019",  # WRA 2019 (Primary Law)
    "../documents_processed/case_precedent"           # Industrial Court Cases
]
# Directory containing the chunks that need filtering
CABINET_CHUNKS_DIR = "../documents_processed/cabinet_chunks_for_filtering"

# Final destination for ALL clean, relevant files (Set to your specified relative path)
RAG_KNOWLEDGE_BASE_DIR = "../rag_knowledge_base"

# Ensure the output directory exists, handling the relative path
if not os.path.exists(RAG_KNOWLEDGE_BASE_DIR):
    os.makedirs(RAG_KNOWLEDGE_BASE_DIR, exist_ok=True)

RELEVANT_LABEL = 1

# **REPLACE THE DICTIONARY BELOW** with the content of  cabinet_training_labels.py file.
CABINET_TRAINING_LABELS = {
    'Highlights_of_Cabinet_Meetingon_Friday07January2022_chunk_7.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_17_June_2022_chunk_7.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_17_June_2022_chunk_18.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_04_April_2025_chunk_8.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Friday_16December_2022_chunk_15.txt': 0,
    'Highlights_of_Cabinet_MeetingonFriday_09_July_2021_chunk_4.txt': 1,
    'Highlights_of_Cabinet_Meetingon_Friday11March2022_chunk_44.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_28_June_2024_chunk_13.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_01_July_2022_chunk_3.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Friday_15_July_2022_chunk_13.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_20_December_2024_chunk_7.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_25_July_2025_chunk_14.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Friday_04_April_2025_chunk_7.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Friday_17November2023_chunk_21.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_31_January_2025_chunk_15.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_10_January_2025_chunk_12.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_29December2023_chunk_4.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Friday_12_July_2024_chunk_29.txt': 0,
    'Highlights_of_Cabinet_Meetingon_Friday15April2022_chunk_19.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_09_August_2024_chunk_1.txt': 1,
    'Highlights_of_Cabinet_Meetingon_Thursday30December2021_chunk_1.txt': 1,
    'Highlights_of_Cabinet_Meetingon_Friday22April2022_chunk_4.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_16December_2022_chunk_2.txt': 1,
    'Highlights_of_Cabinet_Meetingon_Friday01April2022_chunk_21.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_17_January_2025_chunk_18.txt': 0,
    'Highlights_of_Cabinet_MeetingonFriday_26November2021_chunk_10.txt': 1,
    'Highlights_of_Cabinet_Meetingon_Friday11March2022_chunk_5.txt': 0,
    'Highlights_of_Cabinet_MeetingonFriday_26November2021_chunk_9.txt': 1,
    'Highlights_of_Cabinet_Meetingon_Friday25February2022_chunk_26.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_25_July_2025_chunk_51.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_23_May_2025_chunk_10.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_19_May_2023_chunk_17.txt': 0,
    'Highlights_of_Cabinet_Meetingon_Thursday30December2021_chunk_2.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Friday_13_September_2024_chunk_13.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_16December_2022_chunk_1.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Friday_17_January_2025_chunk_9.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Friday_05_August_2022_chunk_30.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_25_July_2025_chunk_13.txt': 1,
    'Highlights_of_Cabinet_Meetingon_Friday07January2022_chunk_8.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Friday_09_August_2024_chunk_2.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Monday_30_January_2023_chunk_3.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_02December_2022_chunk_5.txt': 1,
    'Highlights_of_Cabinet_Meetingon_Friday15April2022_chunk_20.txt': 1,
    'Highlights_of_Cabinet_Meetingon_Friday13May2022_chunk_8.txt': 0,
    'Highlights_of_Cabinet_MeetingonFriday_26November2021_chunk_6.txt': 1,
    'Highlights_of_Cabinet_Meeting_on_Friday_25_July_2025_chunk_8.txt': 0,
    'Highlights_of_Cabinet_Meeting_on_Friday_19_September_2025_chunk_5.txt': 1,
    'Highlights_of_Cabinet_Meetingon_Friday20May2022_chunk_33.txt': 0,
}
# --- END LABELS ---

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def load_text_from_directory(doc_dir):
    """Loads all text files from a directory into a list of documents and filenames."""
    documents = []
    filenames = []
    if os.path.exists(doc_dir):
        for filename in os.listdir(doc_dir):
            if filename.endswith('.txt'):
                try:
                    with open(os.path.join(doc_dir, filename), 'r', encoding='utf-8') as f:
                        documents.append(f.read())
                        filenames.append(filename)
                except Exception as e:
                    print(f"Skipping load of {filename} in {doc_dir}: {e}")
    return documents, filenames

def train_and_filter_cabinet(chunk_docs, chunk_filenames):
    """Trains a classifier on labeled chunks and filters the full chunk set."""
    
    # 1. Prepare Training Data
    training_set = []
    training_labels = []
    
    for filename, label in CABINET_TRAINING_LABELS.items():
        if filename in chunk_filenames:
            idx = chunk_filenames.index(filename)
            training_set.append(chunk_docs[idx])
            training_labels.append(label)

    if len(training_set) < 5:
        print("FATAL ERROR: Insufficient training data for Cabinet filter (less than 5 samples). Cannot proceed.")
        sys.exit(1)

    print(f"\nTraining ML Filter with {len(training_set)} manually labeled chunks...")
    
    # 2. Define and Train ML Pipeline: TF-IDF + Weighted Logistic Regression
    model_pipeline = make_pipeline(
        TfidfVectorizer(stop_words='english', max_df=0.8), 
        # Logistic Regression with class_weight='balanced' is robust
        #LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        
        
        LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    )
    
    # --- DIAGNOSTIC: Evaluate Training Data Quality ---
    # Split the labeled data to test the model's accuracy on unseen samples
    if len(training_set) >= 10: # Only run diagnostics if we have enough samples
        X_train, X_test, y_train, y_test = train_test_split(
            training_set, training_labels, test_size=0.3, random_state=42, stratify=training_labels
        )
        
        # Fit on the training part of the labeled set
        model_pipeline.fit(X_train, y_train)

        # Evaluate on the test part
        y_pred = model_pipeline.predict(X_test)
        
        # Calculate key metrics
        current_precision = precision_score(y_test, y_pred, zero_division=0)
        current_recall = recall_score(y_test, y_pred, zero_division=0)
        
        print(f"\n--- Internal Model Metrics (Diagnostic Test) ---")
        print(f"Test Samples Used: {len(X_test)}")
        print(f"Precision (Accuracy of 'Relevant' predictions): {current_precision:.4f}")
        print(f"Recall (Ability to find all 'Relevant' samples): {current_recall:.4f}")
        print(f"---------------------------------------------")

        # Rerun fit on ALL labeled data to maximize knowledge for the final filter
        model_pipeline.fit(training_set, training_labels)
    else:
        # If too few samples, just train on everything and proceed
        model_pipeline.fit(training_set, training_labels)
    # --- END DIAGNOSTIC ---


    # 3. Filter the Full Chunk Set
    predictions = model_pipeline.predict(chunk_docs)
    
    filtered_chunks = []
    
    for doc_text, filename, prediction in zip(chunk_docs, chunk_filenames, predictions):
        if prediction == RELEVANT_LABEL:
            filtered_chunks.append({'text': doc_text, 'filename': filename, 'source': 'CabinetFiltered'})
            
    return filtered_chunks, len(filtered_chunks)

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # --- STEP 1: LOAD ALREADY RELEVANT DATA (WRA + Cases) ---
    all_relevant_data = []
    initial_relevant_count = 0
    
    for doc_dir in WRA_CASES_DIRS:
        docs, fnames = load_text_from_directory(doc_dir)
        for text, fname in zip(docs, fnames):
            all_relevant_data.append({'text': text, 'filename': fname, 'source': 'CoreLaw'})
        initial_relevant_count += len(docs)
    
    print(f"Initial Core Law documents loaded (WRA + Cases): {initial_relevant_count}")
    
    # --- STEP 2: LOAD CABINET CHUNKS ---
    cabinet_chunks, cabinet_chunk_fnames = load_text_from_directory(CABINET_CHUNKS_DIR)
    initial_chunk_count = len(cabinet_chunks)
    print(f"Cabinet Chunks loaded for filtering: {initial_chunk_count}")

    # --- STEP 3: TRAIN & FILTER CABINET CHUNKS ---
    filtered_cabinet_chunks, filtered_count = train_and_filter_cabinet(cabinet_chunks, cabinet_chunk_fnames)
    
    # --- STEP 4: CONSOLIDATE AND SAVE FINAL KNOWLEDGE BASE ---
    
    all_relevant_data.extend(filtered_cabinet_chunks)
    final_data_df = pd.DataFrame(all_relevant_data)

    saved_count = 0
    for index, row in final_data_df.iterrows():
        output_path = os.path.join(RAG_KNOWLEDGE_BASE_DIR, row['filename'])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(row['text'])
        saved_count += 1
        
    print(f"\n--- Task 2: Document Relevance Filter Complete ---")
    print(f"Total input documents: {initial_relevant_count + initial_chunk_count}")
    print(f"Filtered Cabinet Chunks Kept: {filtered_count}")
    print(f"Total Documents/Chunks in Final Knowledge Base: {saved_count}")
    print(f"The final clean knowledge base is ready in '{RAG_KNOWLEDGE_BASE_DIR}'.")
