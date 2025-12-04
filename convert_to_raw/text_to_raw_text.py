import fitz # PyMuPDF
from docx import Document
import os

# --- Configuration ---
#RAW_FILES_DIRS = ["../documents/cabinet_decisions", "../documents/case_precedent"]
RAW_FILES_DIRS = ["../documents/workers_rights_act_2019"]
OUTPUT_TEXT_DIR = "documents_raw_text"
os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

def extract_text_from_file(file_path):
    """Extracts text based on file extension."""
    text = ""
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.pdf':
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        
        elif file_ext == '.docx':
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + '\n'
                
        else:
            # Skip unrecognized file types
            return None
        
        # Clean up excessive whitespace and newlines
        return ' '.join(text.split())

    except Exception as e:
        print(f"  -> ERROR extracting {os.path.basename(file_path)}: {e}")
        return None

def process_all_files():
    print("--- Starting Text Extraction (Step 2.1) ---")
    extracted_count = 0
    
    for input_dir in RAW_FILES_DIRS:
        # Create a subfolder in the output directory with the same name as the input directory
        output_subfolder = os.path.join(OUTPUT_TEXT_DIR, os.path.basename(input_dir))
        os.makedirs(output_subfolder, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            
            # Skip directories and temporary files
            if os.path.isfile(file_path) and not filename.startswith('.'):
                raw_text = extract_text_from_file(file_path)
                
                if raw_text:
                    # Save the extracted text to the new subfolder
                    output_filename = os.path.splitext(filename)[0] + ".txt"
                    output_path = os.path.join(output_subfolder, output_filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(raw_text)
                    
                    extracted_count += 1
                    # print(f"  -> Extracted and saved: {output_filename}")

    print(f"\nCompleted text extraction. Total text files ready: {extracted_count}")
    return extracted_count

if __name__ == "__main__":
    process_all_files()