import streamlit as st
import cv2
import numpy as np
import img2pdf
from pdf2image import convert_from_bytes
from deskew import determine_skew
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(page_title="PDF Enhancer Tool", layout="centered")

# --- Core Functions ---

def enhance_image(image_cv):
    """
    Method 3: Aggressive Noise Removal + Adaptive Thresholding.
    This mimics the 'Magic Text' filter found in scanner apps.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. Median Blur (Crucial Step)
    # This specific blur is excellent at removing "salt and pepper" noise (the black dots)
    # while keeping text edges sharp.
    denoised = cv2.medianBlur(gray, 5)
    
    # 3. Adaptive Thresholding
    # Block Size (31): Looks at a larger area to understand the background.
    # C (15): High value means we aggressively turn light-gray pixels into white.
    enhanced = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    
    return enhanced

def deskew_image(image_cv):
    """
    Detects the skew angle of the text and rotates the image to straighten it.
    """
    grayscale = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    
    if angle is None or abs(angle) < 0.5:
        return image_cv

    (h, w) = image_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated = cv2.warpAffine(
        image_cv, M, (w, h), flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated

# --- User Interface (UI) ---

st.title("📄 PDF Super Cleaner")
st.write("Upload PDF -> Remove Noise & Whiten Background -> Download")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# 2. Quality Slider
quality = st.slider("Compression Quality (50 is recommended)", 10, 95, 50)

if uploaded_file is not None:
    st.info(f"Filename: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.2f} KB")
    
    if st.button("Start Processing"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            file_bytes = uploaded_file.read()
            
            status_text.text("Reading PDF...")
            pil_images = convert_from_bytes(file_bytes, dpi=150)
            
            processed_images_bytes = []
            total_pages = len(pil_images)
            
            for i, pil_img in enumerate(pil_images):
                progress = int((i / total_pages) * 90)
                progress_bar.progress(progress)
                status_text.text(f"Cleaning Page {i+1} of {total_pages}...")
                
                open_cv_image = np.array(pil_img) 
                open_cv_image = open_cv_image[:, :, ::-1].copy() 

                # Step A: Straighten
                deskewed = deskew_image(open_cv_image)

                # Step B: Clean & Enhance (New Logic)
                enhanced = enhance_image(deskewed)

                # Step C: Compress
                img_pil_final = Image.fromarray(enhanced)
                img_byte_arr = io.BytesIO()
                img_pil_final.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                processed_images_bytes.append(img_byte_arr.getvalue())

            status_text.text("Packing final PDF...")
            final_pdf_bytes = img2pdf.convert(processed_images_bytes)
            
            progress_bar.progress(100)
            status_text.success("Success! Background is clean.")
            
            st.download_button(
                label="📥 Download Clean PDF",
                data=final_pdf_bytes,
                file_name=f"clean_{uploaded_file.name}",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error: {e}")
            if "poppler" in str(e).lower():
                st.warning("System Error: Poppler is not installed on the server.")
