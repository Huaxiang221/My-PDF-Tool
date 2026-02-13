import streamlit as st
import numpy as np
import img2pdf
from pdf2image import convert_from_path, pdfinfo_from_path
from deskew import determine_skew
from PIL import Image
import cv2
import io
import os
import tempfile
import gc
import shutil

# --- Page Config ---
st.set_page_config(page_title="Emergency PDF Tool", layout="centered")

# --- Helper: Resize Image (Crucial for RAM saving) ---
def resize_image_if_too_big(pil_image, max_width=1500):
    """
    If image is wider than max_width, shrink it.
    This prevents the server from running out of RAM.
    """
    width, height = pil_image.size
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return pil_image.resize((max_width, new_height), Image.LANCZOS)
    return pil_image

# --- Helper: Deskew ---
def deskew_image(pil_image):
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

# --- UI ---

st.title("Simple PDF Compressor")
st.write("Higher compression quality = Clearer image, larger file size.")
st.write("Higher pixel = Sharper text (Zoomable), but slower processing.")
st.caption("This tool is made by Seng.")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

st.write("---")
st.subheader("Settings")
auto_straighten = st.checkbox("Auto-Straighten", value=True)
# Lower quality saves disk space
quality = st.slider("Compression Quality", 10, 95, 50) 
# Max Width Slider - The most important setting for crashes
max_width_px = st.select_slider(
    "Max Page Width (Pixels)", 
    options=[1000, 1200, 1500, 2000], 
    value=1200
)
st.caption("Lower pixel width = More stable (No Crash). 1200 is good for reading.")

if uploaded_file is not None:
    st.info(f"File: {uploaded_file.name}")
    
    if st.button("Start Processing"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create a Temporary Folder
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # 1. Save uploaded PDF to Disk first (Don't keep in RAM)
                input_pdf_path = os.path.join(temp_dir, "input.pdf")
                with open(input_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Get Info from Disk
                info = pdfinfo_from_path(input_pdf_path)
                max_pages = int(info["Pages"])
                
                status_text.text(f"Processing {max_pages} pages...")
                saved_image_paths = []

                for i in range(max_pages):
                    # Progress
                    progress = int(((i + 1) / max_pages) * 90)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing Page {i+1}/{max_pages}...")
                    
                    # 3. Read ONE page from Disk
                    # We use convert_from_path which is much better for memory
                    page_images = convert_from_path(
                        input_pdf_path,
                        dpi=150, # Keep DPI low
                        first_page=i+1, 
                        last_page=i+1
                    )
                    
                    if not page_images:
                        continue
                        
                    pil_img = page_images[0]

                    # 4. CRITICAL: Resize immediately!
                    # This reduces the memory footprint by 70-90%
                    pil_img = resize_image_if_too_big(pil_img, max_width=max_width_px)

                    # 5. Deskew (Now safe because image is smaller)
                    if auto_straighten:
                        final_img = deskew_image(pil_img)
                    else:
                        final_img = pil_img

                    # 6. Save to Disk
                    output_img_path = os.path.join(temp_dir, f"page_{i}.jpg")
                    
                    if final_img.mode in ("RGBA", "P"):
                        final_img = final_img.convert("RGB")
                    
                    final_img.save(output_img_path, format='JPEG', quality=quality, optimize=True)
                    saved_image_paths.append(output_img_path)
                    
                    # 7. Cleanup
                    del pil_img
                    del final_img
                    del page_images
                    gc.collect()

                # 8. Build PDF
                status_text.text("Saving final PDF...")
                final_pdf_bytes = img2pdf.convert(saved_image_paths)
                
                progress_bar.progress(100)
                status_text.success("Done!")
                
                st.download_button(
                    label="📥 Download Compressed PDF",
                    data=final_pdf_bytes,
                    file_name=f"Processed_{uploaded_file.name}",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Error: {e}")
            if "poppler" in str(e).lower():
                st.warning("System Error: Poppler missing.")

