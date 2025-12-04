import os
from langchain_text_splitters import RecursiveCharacterTextSplitter  

# --- Configuration ---
CABINET_DECISIONS_DIR = "../documents_processed/documents_raw_text/cabinet_decisions"
# Output directory for the individual, chunked documents
CHUNKED_DOCS_DIR = "../documents_processed/cabinet_chunks_for_filtering"

os.makedirs(CHUNKED_DOCS_DIR, exist_ok=True)
CHUNK_SIZE = 500    # Optimal size for a single, focused piece of information
CHUNK_OVERLAP = 50  # Overlap to maintain context between chunks

def load_and_chunk_documents():
    """Loads Cabinet text files, splits them into small chunks, and saves each chunk."""
    
    # We use LangChain's Recursive splitter, which is the standard tool for RAG chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # Corrected syntax: This is the priority list for splitting
        separators=["\n\n", "\n", " ", ""] 
    )
    
    total_chunks = 0
    print(f"--- Starting Chunking of Cabinet Decisions (Task 2.A) ---")
    print(f"Reading files from: {CABINET_DECISIONS_DIR}")

    if os.path.exists(CABINET_DECISIONS_DIR):
        for filename in os.listdir(CABINET_DECISIONS_DIR):
            # We assume you have run the text extraction step and have .txt files
            if filename.endswith('.txt'): 
                file_path = os.path.join(CABINET_DECISIONS_DIR, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        
                    # 1. Split the document into chunks
                    chunks = text_splitter.split_text(text)
                    
                    # 2. Save each chunk as a new file for easy labeling/filtering
                    for i, chunk in enumerate(chunks):
                        # Creates a unique name like 'Highlights_of_Cabinet_Meeting_chunk_1.txt'
                        base_name = os.path.splitext(filename)[0]
                        chunk_filename = f"{base_name}_chunk_{i+1}.txt"
                        output_path = os.path.join(CHUNKED_DOCS_DIR, chunk_filename)
                        
                        with open(output_path, 'w', encoding='utf-8') as f_out:
                            f_out.write(chunk)
                            total_chunks += 1
                            
                    # print(f"  -> Chunked {filename} into {len(chunks)} parts.")
                        
                except Exception as e:
                    print(f"ERROR processing {filename}: {e}")

    print(f"\nCompleted Chunking. Total chunks created: {total_chunks}")
    print(f"Chunks are saved in the '{CHUNKED_DOCS_DIR}' folder, ready for labeling.")
    return total_chunks

if __name__ == "__main__":
    load_and_chunk_documents()