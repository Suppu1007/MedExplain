
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

import easyocr
import fitz  # PyMuPDF
from fastapi import UploadFile, HTTPException
from PIL import Image
import numpy as np

# Initialize EasyOCR reader lazily
reader = None

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'])
    return reader


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def save_upload_file(upload_file: UploadFile) -> Path:
    try:
        file_path = UPLOAD_DIR / upload_file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

def extract_text_from_image(image_path: str) -> str:
    try:
        result = get_reader().readtext(image_path, detail=0)
        return "\n".join(result)
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    text_content = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            # 1. Try extracting embedded text
            text = page.get_text()
            
            # 2. If little text (scanned PDF), use OCR on page image
            if len(text.strip()) < 50:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_array = np.array(img)
                ocr_text = get_reader().readtext(img_array, detail=0)
                text = "\n".join(ocr_text)
            
            text_content.append(text)
            
        return "\n\n".join(text_content)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

async def process_document(file: UploadFile) -> dict:
    file_path = save_upload_file(file)
    extracted_text = ""
    
    extension = file_path.suffix.lower()
    
    if extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        extracted_text = extract_text_from_image(str(file_path))
    elif extension == '.pdf':
        extracted_text = extract_text_from_pdf(str(file_path))
    else:
        # Clean up and error
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Unsupported file type")
        
    return {
        "filename": file.filename,
        "file_path": str(file_path),
        "extracted_text": extracted_text
    }
