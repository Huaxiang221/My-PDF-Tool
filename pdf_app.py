import streamlit as st
import numpy as np
import img2pdf
from pdf2image import convert_from_bytes
from deskew import determine_skew
from PIL import Image
import cv2
import io
import os
import tempfile
import gc  # Garbage collection (Memory cleanup)

# --- Page Config ---
st.set_page_config(page_title="Stable PDF Compressor", layout="centered")

# --- Helper Function: Straighten Image ---
def deskew_image(pil_image):
    """
    Detects skew angle and rotates the image to straighten it.
    """
    open_cv_image = np.array(pil_image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    angle = determine_skew(gray)
    
    if angle is None or abs(angle) < 0.5:
        return pil_image

    (h, w) = open_cv_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated = cv2.warpAffine(
        open_cv_image, M, (w, h), flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return Image.fromarray(rotated)

# --- User Interface ---

st.title("🛡️ Anti-Crash PDF Compressor")
st.write("Safe Mode: Processes pages one by one to prevent server crashes.")
st.caption("Perfect for large files or high-resolution scans.")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

st.write("---")
st.subheader("Settings")
auto_straighten = st.checkbox("Auto-Straighten (Fix Crooked Pages)", value=True)
quality = st.slider("Compression Quality (Lower = Smaller File)", 10, 95, 50)

if uploaded_file is not None:
    st.info(f"File Name: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.2f} KB")
    
    if st.button("Start Safe Processing"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create a temporary directory to store processed images
            # This prevents running out of RAM (Memory)
            with tempfile.TemporaryDirectory() as temp_dir:
                file_bytes = uploaded_file.read()
                status_text.text("Reading PDF...")
                
                # Convert PDF to images (200 DPI is a good balance)
                pil_images = convert_from_bytes(file_bytes, dpi=200)
                
                total_pages = len(pil_images)
                saved_image_paths = []

                for i, pil_img in enumerate(pil_images):
                    # Update Progress
                    progress = int((i / total_pages) * 90)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing Page {i+1}/{total_pages}...")
                    
                    # 1. Straighten (Optional)
                    if auto_straighten:
                        final_img = deskew_image(pil_img)
                    else:
                        final_img = pil_img

                    # 2. Save immediately to Disk (Temporary Folder)
                    # We save the file path instead of keeping the image in memory
                    temp_img_path = os.path.join(temp_dir, f"page_{i}.jpg")
                    
                    # Ensure RGB mode for JPEG saving
                    if final_img.mode in ("RGBA", "P"):
                        final_img = final_img.convert("RGB")
                    
                    final_img.save(temp_img_path, format='JPEG', quality=quality, optimize=True)
                    saved_image_paths.append(temp_img_path)
                    
                    # --- CRITICAL STEP: Free up Memory ---
                    del final_img  # Delete processed image variable
                    del pil_img    # Delete original image variable
                    gc.collect()   # Force Python to clean up memory immediately

                # 3. Build PDF from Disk
                status_text.text("Combining pages into PDF...")
                
                # img2pdf can read directly from file paths (Very memory efficient)
                final_pdf_bytes = img2pdf.convert(saved_image_paths)
                
                progress_bar.progress(100)
                status_text.success("Success! Processed without crashing.")
                
                st.download_button(
                    label="📥 Download Compressed PDF",
                    data=final_pdf_bytes,
                    file_name=f"Compressed_{uploaded_file.name}",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Error: {e}")
            if "Memory" in str(e):
                st.warning("⚠️ The file is still too large. Try splitting the PDF into smaller parts.")
            elif "poppler" in str(e).lower():
                st.warning("System Error: Poppler is not installed on the server.")
