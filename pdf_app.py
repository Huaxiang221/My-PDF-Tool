import streamlit as st
import numpy as np
import img2pdf
from pdf2image import convert_from_bytes
from deskew import determine_skew
from PIL import Image
import cv2
import io

st.set_page_config(page_title="Simple PDF Compressor", layout="centered")

# --- Helper Function: Only Rotate, Don't Change Color ---
def deskew_image(pil_image):
    """
    Detects skew angle and rotates the image.
    Does NOT change colors.
    """
    # Convert PIL to OpenCV (RGB)
    open_cv_image = np.array(pil_image)
    
    # Create a grayscale copy just to calculate the angle
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    angle = determine_skew(gray)
    
    # If angle is very small, return original image
    if angle is None or abs(angle) < 0.5:
        return pil_image

    # Calculate rotation
    (h, w) = open_cv_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the original RGB image
    # borderValue=(255, 255, 255) means fill corners with White
    rotated = cv2.warpAffine(
        open_cv_image, M, (w, h), flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    
    # Convert back to PIL
    return Image.fromarray(rotated)

# --- UI ---

st.title("📂 Simple PDF Compressor")
st.write("Compress PDF file size while keeping ORIGINAL colors.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Settings
st.write("---")
st.subheader("Settings")
# 1. Straighten Option
auto_straighten = st.checkbox("Auto-Straighten (Fix Crooked Pages)", value=True)
# 2. Compression Slider
quality = st.slider("Compression Level (Lower = Smaller File)", 10, 95, 50)
st.caption("Note: 50 is a good balance. 20 is very small but blurry.")

if uploaded_file is not None:
    st.info(f"Original File: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.2f} KB")
    
    if st.button("Start Compression"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            file_bytes = uploaded_file.read()
            status_text.text("Reading PDF...")
            
            # Convert PDF to Images (200 DPI is standard for reading)
            pil_images = convert_from_bytes(file_bytes, dpi=200)
            
            processed_images_bytes = []
            total_pages = len(pil_images)
            
            for i, pil_img in enumerate(pil_images):
                progress = int((i / total_pages) * 90)
                progress_bar.progress(progress)
                status_text.text(f"Processing Page {i+1}/{total_pages}...")
                
                # 1. Straighten (Optional)
                if auto_straighten:
                    final_img = deskew_image(pil_img)
                else:
                    final_img = pil_img

                # 2. Compress & Save
                # We save directly as JPEG using PIL. 
                # This keeps all colors exactly as they are, just compresses the data.
                img_byte_arr = io.BytesIO()
                
                # Convert to RGB (in case of PNG/Transparent inputs) to save as JPEG
                if final_img.mode in ("RGBA", "P"):
                    final_img = final_img.convert("RGB")
                    
                final_img.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                processed_images_bytes.append(img_byte_arr.getvalue())

            status_text.text("Building PDF...")
            final_pdf_bytes = img2pdf.convert(processed_images_bytes)
            
            progress_bar.progress(100)
            status_text.success("Done! Colors preserved.")
            
            st.download_button(
                label="📥 Download Compressed PDF",
                data=final_pdf_bytes,
                file_name=f"Compressed_{uploaded_file.name}",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error: {e}")
            if "poppler" in str(e).lower():
                st.warning("System Hint: Poppler is not installed on the server.")
