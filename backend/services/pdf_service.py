import io
import os
import cv2
import numpy as np
import pdfplumber
import easyocr
from PIL import Image, ImageOps, ImageFilter
from pdf2image import convert_from_path

class PDFService:
    # Initialize EasyOCR Reader as a class attribute (loads model into memory once)
    # Using 'en' for English medical terms
    _reader = None

    @classmethod
    def get_reader(cls):
        if cls._reader is None:
            print("Loading EasyOCR Engine...")
            cls._reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if CUDA is available
        return cls._reader

    @staticmethod
    def _enhance_image(image: Image.Image) -> Image.Image:
        """Balanced clinical image enhancement for AI-driven OCR (EasyOCR)."""
        # 1. Convert to OpenCv format
        img_cv = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # 2. Edge-Preserving Denoising (Bilateral Filter)
        # Much better than Median Blur for text as it keeps character edges sharp
        denoised = cv2.bilateralFilter(img_cv, 9, 75, 75)
        
        # 3. Grayscale conversion
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        # 4. Contrast Enhancement (CLAHE)
        # Local contrast enhancement to help text stand out without binarization artifacts
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 5. Convert back to PIL
        image = Image.fromarray(enhanced)
        
        # 6. Upscale image to improve small text recognition (2x)
        w, h = image.size
        image = image.resize((w*2, h*2), Image.Resampling.LANCZOS)
        
        return image

    @staticmethod
    def extract_text(file_bytes: bytes, filename: str) -> str:
        """
        Capstone-Grade Extraction:
        Handles Searchable PDFs, Scanned PDFs, and Images.
        """
        text_content = []
        filename_lower = filename.lower()
        
        try:
            # --- CASE 1: PDF HANDLING ---
            if filename_lower.endswith(".pdf"):
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        
                        # If text is found (Searchable PDF)
                        if page_text and len(page_text.strip()) > 20:
                            print(f"[{filename}] Page {page.page_number}: Searchable text found.")
                            text_content.append(page_text)
                        else:
                            # If no text found, it's likely a scan. OCR the page.
                            print(f"[{filename}] Page {page.page_number} appears to be a scan. Initializing OCR...")
                            
                            # 1. Try Enhanced OCR
                            page_image = page.to_image(resolution=300).original
                            enhanced_img = PDFService._enhance_image(page_image)
                            
                            results = PDFService.get_reader().readtext(
                                np.array(enhanced_img), 
                                detail=0,
                                paragraph=True # Better context grouping
                            )
                            
                            # 2. Fallback: Raw OCR
                            if not results or len(" ".join(results).strip()) < 5:
                                print(f"[{filename}] Page {page.page_number}: Enhanced OCR failed. Retrying with RAW...")
                                results = PDFService.get_reader().readtext(
                                    np.array(page_image.convert('RGB')), 
                                    detail=0,
                                    paragraph=True
                                )
                            
                            page_final = " ".join(results)
                            
                            # 3. Ultimate Fallback: Vision LLM (Moondream)
                            if not results or len(page_final.strip()) < 5:
                                print(f"[{filename}] Page {page.page_number}: EasyOCR failed. Triggering Vision AI...")
                                from services.llm import stream_vision_response
                                v_prompt = "Carefully transcribe all clinical text and values in this lab report. Output ONLY the extracted text."
                                
                                # Convert page to bytes for VLM
                                img_byte_arr = io.BytesIO()
                                page_image.save(img_byte_arr, format='PNG')
                                vlm_text = ""
                                for token in stream_vision_response(v_prompt, img_byte_arr.getvalue()):
                                    vlm_text += token
                                
                                if len(vlm_text.strip()) > 10:
                                    print(f"[{filename}] Page {page.page_number}: Vision AI successfully extracted text.")
                                    page_final = vlm_text
                            
                            text_content.append(page_final)

            # --- CASE 2: IMAGE HANDLING (OCR) ---
            elif filename_lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
                image = Image.open(io.BytesIO(file_bytes))
                enhanced_img = PDFService._enhance_image(image)
                
                # Use EasyOCR with paragraph grouping for better context
                results = PDFService.get_reader().readtext(
                    np.array(enhanced_img), 
                    detail=0, 
                    paragraph=True
                )
                
                # Robustness Fallback: If enhanced image yielded nothing, try the RAW image
                if not results or len("\n".join(results).strip()) < 10:
                    print(f"[{filename}] Enhanced OCR failed. Retrying with RAW image...")
                    results = PDFService.get_reader().readtext(
                        np.array(image.convert('RGB')), 
                        detail=0, 
                        paragraph=True
                    )
                
                # Ultimate Fallback: Vision LLM
                img_final = "\n".join(results)
                if not results or len(img_final.strip()) < 10:
                    print(f"[{filename}] EasyOCR failed. Triggering Vision AI...")
                    from services.llm import stream_vision_response
                    v_prompt = "Examine this clinical document carefully. Transcribe all text, numbers, and biomarkers exactly as they appear."
                    vlm_text = ""
                    for token in stream_vision_response(v_prompt, file_bytes):
                        vlm_text += token
                    if len(vlm_text.strip()) > 10:
                        img_final = vlm_text
                
                text_content.append(img_final)

            else:
                print(f"Unsupported format: {filename}")
                return ""

            # Final Cleanup: Join all pages and remove artifacts
            final_text = "\n".join(text_content).strip()
            
            # Print debug snippet for Capstone logging
            print(f"Extraction Successful: {filename} ({len(final_text)} chars)")
            return final_text

        except Exception as e:
            print(f"Clinical Extraction Error [{filename}]: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""