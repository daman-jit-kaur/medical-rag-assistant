import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path to your data folder
DATA_FOLDER = "data"
OUTPUT_FOLDER = "extracted_texts"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract all text from a single PDF file."""
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {page_num + 1} ---\n"
            full_text += text
    
    return full_text

def process_all_pdfs():
    """Loop through all PDFs in data folder and extract text."""
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in data folder!")
        return
    
    print(f"Found {len(pdf_files)} PDF files. Starting extraction...\n")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_FOLDER, pdf_file)
        print(f"Processing: {pdf_file}")
        
        try:
            text = extract_text_from_pdf(pdf_path)
            
            # Save extracted text to a .txt file
            output_filename = pdf_file.replace(".pdf", ".txt")
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"✅ Saved: {output_filename} ({len(text)} characters)\n")
        
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}\n")
    
    print("Extraction complete!")
    print(f"All text files saved in: {OUTPUT_FOLDER}/")

# Run the extraction
if __name__ == "__main__":
    process_all_pdfs()