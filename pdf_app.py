import streamlit as st
import numpy as np
import img2pdf
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from deskew import determine_skew
from PIL import Image
import cv2
import io
import os
import tempfile
import gc  # Garbage Collection for memory cleanup

# --- Page Config ---
st.set_page_config(page_title="Ultra-Stable PDF Tool", layout="centered")

# --- Helper Function: Straighten Image ---
def deskew_image(pil_image):
    """
    Detects skew angle and rotates the image to straighten it.
    """
    open_cv_image = np.array(pil_image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    angle = determine_skew(gray)
    
    # If angle is very small, return original image
    if angle is None or abs(angle) < 0.5:
        return pil_image

    (h, w) = open_cv_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate with white background fill
    rotated = cv2.warpAffine(
        open_cv_image, M, (w, h), flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return Image.fromarray(rotated)

# --- User Interface ---

st.title("🛡️ Ultra-Stable PDF Compressor")
st.write("Chunk Processing Mode: Reads 1 page at a time. Zero RAM spikes.")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

st.write("---")
st.subheader("Settings")
auto_straighten = st.checkbox("Auto-Straighten (Fix Crooked Pages)", value=True)
quality = st.slider("Compression Quality (Lower = Smaller File)", 10, 95, 50)
dpi_level = st.selectbox("Resolution (DPI)", [150, 200], index=0) 
st.caption("150 DPI is faster and safer for mobile. 200 is better for printing.")

if uploaded_file is not None:
    st.info(f"File Name: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.2f} KB")
    
    if st.button("Start Processing"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Use a temporary directory to store page images
            with tempfile.TemporaryDirectory() as temp_dir:
                file_bytes = uploaded_file.read()
                
                # 1. Critical Step: Get PDF info ONLY (Do not read images yet)
                # This is instant and takes almost zero memory
                info = pdfinfo_from_bytes(file_bytes)
                max_pages = int(info["Pages"])
                
                status_text.text(f"Detected {max_pages} pages. Starting chunk processing...")
                
                saved_image_paths = []

                # 2. Process page by page
                for i in range(max_pages):
                    # Update Progress Bar
                    progress = int(((i + 1) / max_pages) * 90)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing Page {i+1}/{max_pages}...")
                    
                    # 3. Critical Step: Convert ONLY the current page
                    # 'first_page' and 'last_page' ensure we only load 1 page into RAM
                    page_images = convert_from_bytes(
                        file_bytes, 
                        dpi=dpi_level, 
                        first_page=i+1, 
                        last_page=i+1
                    )
                    
                    if not page_images:
                        continue
                        
                    pil_img = page_images[0]

                    # 4. Straighten (Optional)
                    if auto_straighten:
                        final_img = deskew_image(pil_img)
                    else:
                        final_img = pil_img

                    # 5. Save immediately to Disk
                    temp_img_path = os.path.join(temp_dir, f"page_{i}.jpg")
                    
                    # Ensure RGB mode
                    if final_img.mode in ("RGBA", "P"):
                        final_img = final_img.convert("RGB")
                    
                    final_img.save(temp_img_path, format='JPEG', quality=quality, optimize=True)
                    saved_image_paths.append(temp_img_path)
                    
                    # 6. Aggressive Memory Cleanup
                    del pil_img
                    del final_img
                    del page_images
                    gc.collect()

                # 7. Build Final PDF
                status_text.text("Merging pages into PDF...")
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
            if "poppler" in str(e).lower():
                st.warning("System Error: Poppler is not installed on the server.")
