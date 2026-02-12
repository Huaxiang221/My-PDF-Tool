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
    New Method: Strong Denoising + Otsu Binarization.
    This creates a clean 'scanned document' look (Pure Black & White).
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur (Important!)
    # This step smooths out the "pepper" noise/dots from the paper background.
    # (5, 5) is the kernel size. If still noisy, you can try (7, 7).
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Otsu's Binarization
    # This algorithm automatically calculates the best threshold to separate text from background.
    # It results in sharp black text on a purely white background.
    ret, enhanced = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return enhanced

def deskew_image(image_cv):
    """
    Detects the skew angle of the text and rotates the image to straighten it.
    """
    grayscale = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    
    # If the angle is very small, we don't need to rotate it
    if angle is None or abs(angle) < 0.5:
        return image_cv

    (h, w) = image_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate and fill the background with white
    rotated = cv2.warpAffine(
        image_cv, M, (w, h), flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated

# --- User Interface (UI) ---

st.title("📄 PDF Scanner & Enhancer")
st.write("Upload PDF -> Auto Straighten, Clean & Compress -> Download")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# 2. Quality Slider
quality = st.slider("Compression Quality (Lower = Smaller File Size)", 10, 95, 50)

if uploaded_file is not None:
    # Show file info
    st.info(f"Filename: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.2f} KB")
    
    if st.button("Start Processing"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Read file from memory
            file_bytes = uploaded_file.read()
            
            status_text.text("Converting PDF pages to images...")
            # dpi=150 is good enough for mobile viewing and faster processing
            pil_images = convert_from_bytes(file_bytes, dpi=150)
            
            processed_images_bytes = []
            total_pages = len(pil_images)
            
            for i, pil_img in enumerate(pil_images):
                # Update progress
                progress = int((i / total_pages) * 90)
                progress_bar.progress(progress)
                status_text.text(f"Processing Page {i+1} of {total_pages}...")
                
                # Convert PIL to OpenCV format
                open_cv_image = np.array(pil_img) 
                open_cv_image = open_cv_image[:, :, ::-1].copy() 

                # Step A: Straighten (Deskew)
                deskewed = deskew_image(open_cv_image)

                # Step B: Enhance (Clean Black & White)
                enhanced = enhance_image(deskewed)

                # Step C: Compress
                img_pil_final = Image.fromarray(enhanced)
                img_byte_arr = io.BytesIO()
                # Optimize=True and Quality controls the file size
                img_pil_final.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                processed_images_bytes.append(img_byte_arr.getvalue())

            # Final Step: Combine into PDF
            status_text.text("Generating final PDF file...")
            final_pdf_bytes = img2pdf.convert(processed_images_bytes)
            
            progress_bar.progress(100)
            status_text.success("Done! Your file is ready.")
            
            # Download Button
            st.download_button(
                label="📥 Download Enhanced PDF",
                data=final_pdf_bytes,
                file_name=f"enhanced_{uploaded_file.name}",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error occurred: {e}")
            if "poppler" in str(e).lower():
                st.warning("System Error: Poppler is not installed on the server.")
