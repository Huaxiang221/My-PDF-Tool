import streamlit as st
import cv2
import numpy as np
import img2pdf
from pdf2image import convert_from_bytes
from deskew import determine_skew
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(page_title="Color PDF Cleaner", layout="centered")

# --- Core Processing Functions ---

def enhance_image_color(image_cv):
    """
    Retains color (e.g., red/blue pen) while removing light background noise.
    """
    # 1. Slight Gaussian Blur to smooth out paper noise
    blurred = cv2.GaussianBlur(image_cv, (3, 3), 0)
    
    # 2. Color Cleaning (White Balancing / Thresholding)
    # Convert to HSV to check the 'Value' (Brightness) channel
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Create a mask: pixels brighter than 200 are considered "background"
    # This detects light gray paper, shadows, and faint bleed-through text
    mask = v > 200 
    
    # 3. Boost Contrast and Saturation
    # alpha > 1.0 (Increase contrast), beta < 0 (Darken the darks)
    # This makes the ink look deeper and the background cleaner
    enhanced = cv2.convertScaleAbs(blurred, alpha=1.4, beta=-30)
    
    # 4. Apply the "Whitening" Mask
    # Force the detected background pixels to become pure white (255, 255, 255)
    enhanced[mask] = [255, 255, 255]
    
    # 5. Sharpening
    # Makes the colored text edges crisper
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    final_img = cv2.filter2D(enhanced, -1, kernel)
    
    return final_img

def deskew_image(image_cv):
    """
    Detects skew angle and rotates the image (calculated on grayscale, applied to color).
    """
    # Convert to grayscale just for angle detection
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(gray)
    
    # If angle is negligible, return original
    if angle is None or abs(angle) < 0.5:
        return image_cv

    (h, w) = image_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the COLOR image, filling background with white
    rotated = cv2.warpAffine(
        image_cv, M, (w, h), flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated

# --- User Interface (UI) ---

st.title("🎨 Color PDF Cleaner")
st.write("Upload -> Keep Red/Blue Notes + Remove Noise -> Download")
st.caption("Best for exam papers, marked homework, or documents with stamps.")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
quality = st.slider("Output Quality (Recommended 80-90)", 50, 100, 85)

if uploaded_file is not None:
    st.info(f"File: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.2f} KB")
    
    if st.button("Start Color Processing"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            file_bytes = uploaded_file.read()
            
            # Using 250 DPI for a good balance of speed and color detail
            status_text.text("Reading PDF (250 DPI)...")
            pil_images = convert_from_bytes(file_bytes, dpi=250)
            
            processed_images_bytes = []
            total_pages = len(pil_images)
            
            for i, pil_img in enumerate(pil_images):
                progress = int((i / total_pages) * 90)
                progress_bar.progress(progress)
                status_text.text(f"Processing Page {i+1}/{total_pages}...")
                
                # Convert PIL to OpenCV format (BGR)
                open_cv_image = np.array(pil_img) 
                open_cv_image = open_cv_image[:, :, ::-1].copy() 

                # 1. Straighten (Deskew)
                deskewed = deskew_image(open_cv_image)

                # 2. Color Enhance & Clean
                enhanced = enhance_image_color(deskewed)

                # 3. Save
                # Convert back to RGB for PIL
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                img_pil_final = Image.fromarray(enhanced_rgb)
                
                img_byte_arr = io.BytesIO()
                img_pil_final.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                processed_images_bytes.append(img_byte_arr.getvalue())

            status_text.text("Packing final PDF...")
            final_pdf_bytes = img2pdf.convert(processed_images_bytes)
            
            progress_bar.progress(100)
            status_text.success("Done! Colors preserved, background whitened.")
            
            st.download_button(
                label="📥 Download Cleaned PDF",
                data=final_pdf_bytes,
                file_name=f"Color_Clean_{uploaded_file.name}",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error occurred: {e}")
            if "poppler" in str(e).lower():
                st.warning("System Hint: Poppler is missing on the server.")
