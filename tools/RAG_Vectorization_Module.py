import os
import sys
from dotenv import load_dotenv

# Import core RAG components
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# NEW: Import Hugging Face components for BERT Embeddings
from langchain_huggingface import HuggingFaceEmbeddings 

# --- Configuration (Based on your project structure) ---
RAG_KNOWLEDGE_BASE_DIR = "../rag_knowledge_base"
CHROMA_DB_PATH = "../chroma_db_index"

 #Ensure the output directory exists
if not os.path.exists(CHROMA_DB_PATH):
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# --- RAG Parameters ---
BERT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # A fast, high-quality BERT variant
CHUNK_SIZE = 1000  # Optimal chunk size for embedding models
CHUNK_OVERLAP = 200 # Overlap ensures context is preserved across splits

def setup_and_index_db():
    print("--- Starting Task 3: Vectorization and Indexing (BERT Model) ---")
    
    # --- 1. Load Documents from Final Knowledge Base ---
    print(f"\nLoading documents from: {RAG_KNOWLEDGE_BASE_DIR}")
    
    # Use DirectoryLoader to fetch all .txt files in the knowledge base
    loader = DirectoryLoader(
        RAG_KNOWLEDGE_BASE_DIR, 
        glob="**/*.txt", 
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    try:
        # LangChain loads them as a list of Document objects
        documents = loader.load()
        if not documents:
            print(f"FATAL ERROR: No documents loaded. Is the directory '{RAG_KNOWLEDGE_BASE_DIR}' correct and full of .txt files?")
            sys.exit(1)
        print(f"Loaded {len(documents)} documents/chunks from the knowledge base.")
    except Exception as e:
        print(f"FATAL ERROR: Document loading failed. Error: {e}")
        sys.exit(1)

    # --- 2. Chunking (Final Step for WRA and Case Law files) ---
    print("\nChunking large documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # The splitter processes both large documents (WRA/Cases) and the already small Cabinet chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Total uniform chunks created for indexing: {len(chunks)}")
    
    # --- 3. Embedding and Indexing using BERT (Hugging Face) ---
    print(f"\nCreating BERT Embeddings using model: {BERT_EMBEDDING_MODEL}")
    
    # Initialize the BERT-based embedding model
    # Note: This step downloads the model weights if they are not already cached.
    embeddings = HuggingFaceEmbeddings(
        model_name=BERT_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} # Set to 'cuda' if you have a GPU
    )

    # Create the vector store index and persist to disk
    print(f"Storing vectors in ChromaDB at: {CHROMA_DB_PATH}")
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_DB_PATH
    )
    
    # Persist the database to disk
    db.persist()
    
    print("\n" + "="*50)
    print("TASK 3 COMPLETE: VECTOR DATABASE CREATED")
    print(f"Total indexed chunks: {len(chunks)}")
    print(f"ChromaDB saved to: {CHROMA_DB_PATH}")
    print("="*50)

if __name__ == "__main__":
    setup_and_index_db() 