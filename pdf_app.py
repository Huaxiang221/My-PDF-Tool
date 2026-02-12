import streamlit as st
import cv2
import numpy as np
import img2pdf
from pdf2image import convert_from_bytes
from deskew import determine_skew
from PIL import Image
import io

st.set_page_config(page_title="PDF Cleaner Pro", layout="centered")

# --- Core Logic Functions ---

def remove_handwriting_logic(image_cv, threshold_val):
    """
    Attempts to remove handwriting.
    Principle: Printed text is usually pure black (value ~0), while handwriting
    is often gray or colored (value ~50-150).
    We use a threshold to turn everything not "black enough" into white.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur to smooth out pen strokes texture
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Thresholding
    # Any pixel BRIGHTER than threshold_val becomes WHITE (255).
    # Any pixel DARKER than threshold_val becomes BLACK (0).
    # Lower threshold = Aggressive removal (only keeps pitch black text).
    _, binary = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    
    return binary

def enhance_image_color(image_cv):
    """
    Color Enhance Mode: Keeps red/blue marks but whitens the background.
    """
    blurred = cv2.GaussianBlur(image_cv, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Detect light background (>200 brightness)
    mask = v > 200 
    
    # Increase contrast
    enhanced = cv2.convertScaleAbs(blurred, alpha=1.4, beta=-30)
    
    # Apply white mask to background
    enhanced[mask] = [255, 255, 255]
    
    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    final_img = cv2.filter2D(enhanced, -1, kernel)
    return final_img

def deskew_image(image_cv):
    """
    Straightens the image.
    """
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(gray)
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

st.title("🧼 Magic PDF Eraser")
st.write("Upload -> Remove Handwriting or Enhance Color -> Download")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

# --- Sidebar Settings ---
st.sidebar.header("Settings")

# Mode Selection
mode = st.sidebar.radio(
    "Select Mode:",
    ("🎨 Color Enhance (Keep Color)", "✏️ Handwriting Remover (Black & White)")
)

if mode == "✏️ Handwriting Remover (Black & White)":
    st.sidebar.info("💡 **Tip:** Lower the threshold to remove more handwriting. Raise it if printed text is disappearing.")
    
    # Slider for Handwriting Removal
    # Default is 100. 
    # Move LEFT (e.g., 80) to remove more gray/light text.
    # Move RIGHT (e.g., 120) to keep more text.
    remove_threshold = st.sidebar.slider("Eraser Threshold", 50, 180, 100)
    quality = 80 
else:
    remove_threshold = None
    quality = st.sidebar.slider("Output Quality", 50, 100, 85)


if uploaded_file is not None:
    st.info(f"File: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.2f} KB")
    
    if st.button("Start Processing"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            file_bytes = uploaded_file.read()
            status_text.text("Reading PDF (200 DPI)...")
            
            # Using 200 DPI for speed
            pil_images = convert_from_bytes(file_bytes, dpi=200)
            
            processed_images_bytes = []
            total_pages = len(pil_images)
            
            for i, pil_img in enumerate(pil_images):
                progress = int((i / total_pages) * 90)
                progress_bar.progress(progress)
                status_text.text(f"Processing Page {i+1}/{total_pages}...")
                
                # Convert to OpenCV
                open_cv_image = np.array(pil_img) 
                open_cv_image = open_cv_image[:, :, ::-1].copy() 

                # 1. Straighten
                deskewed = deskew_image(open_cv_image)

                # 2. Apply chosen Logic
                if mode == "🎨 Color Enhance (Keep Color)":
                    final_img = enhance_image_color(deskewed)
                    # Convert BGR to RGB
                    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                else:
                    # Handwriting Removal Mode
                    final_img = remove_handwriting_logic(deskewed, remove_threshold)
                    # Convert Gray to RGB
                    final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)

                # 3. Save
                img_pil_final = Image.fromarray(final_img)
                img_byte_arr = io.BytesIO()
                img_pil_final.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                processed_images_bytes.append(img_byte_arr.getvalue())

            status_text.text("Packing final PDF...")
            final_pdf_bytes = img2pdf.convert(processed_images_bytes)
            
            progress_bar.progress(100)
            status_text.success("Done!")
            
            st.download_button(
                label="📥 Download Result",
                data=final_pdf_bytes,
                file_name=f"Processed_{uploaded_file.name}",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error: {e}")
            if "poppler" in str(e).lower():
                st.warning("System Hint: Poppler is not installed on the server.")
