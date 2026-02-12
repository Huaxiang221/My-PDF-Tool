import streamlit as st
import cv2
import numpy as np
import img2pdf
from pdf2image import convert_from_bytes
from deskew import determine_skew
from PIL import Image, ImageEnhance
import io

# --- 页面配置 ---
st.set_page_config(page_title="High-Res PDF Enhancer", layout="centered")

# --- 核心处理函数 (高清版) ---

def enhance_image(image_cv):
    """
    Method: High-Fidelity Enhancement.
    Keeps the text smooth (anti-aliased) while whitening the background.
    """
    # 1. 转为灰度
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. 增强对比度 (让黑的更黑，白的更白，但保留中间的过渡)
    # 这一步代替了暴力的“二值化”，所以字不会有锯齿
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 3. 简单的亮度调整，确保背景是纯白
    # 任何亮于 200 的灰色都会变成纯白 255
    _, result = cv2.threshold(enhanced, 200, 255, cv2.THRESH_TRUNC)
    
    # 4. 反转颜色恢复 (因为 TRUNC 会变暗，我们需要重新拉伸直方图)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    
    # 5. 最后一道保险：把浅灰色背景彻底变白，保留深色文字
    # 这是一个平滑的阈值处理
    result = cv2.convertScaleAbs(result, alpha=1.2, beta=10) # 增加对比度
    
    # 稍微做一点点模糊来平滑噪点，但非常轻微
    result = cv2.GaussianBlur(result, (3, 3), 0)
    
    return result

def deskew_image(image_cv):
    """
    Detect skew and rotate.
    """
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(gray)
    
    if angle is None or abs(angle) < 0.5:
        return image_cv

    (h, w) = image_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 旋转时使用白色填充背景
    rotated = cv2.warpAffine(
        image_cv, M, (w, h), flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated

# --- 界面 ---

st.title("🔍 HD PDF Scanner (高清版)")
st.write("Upload -> High Quality Process (300 DPI) -> Download")
st.caption("Note: Processing is slower because the quality is much higher.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# 默认质量设为 85，保证清晰度
quality = st.slider("Output Quality (Keep it high for clear text)", 50, 100, 85)

if uploaded_file is not None:
    st.info(f"File: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.2f} KB")
    
    if st.button("Start HD Processing"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            file_bytes = uploaded_file.read()
            
            status_text.text("Scanning at 300 DPI (High Res)... Please wait...")
            
            # --- 关键修改：DPI 改为 300 ---
            # 这会使处理时间变长，但清晰度大大增加
            pil_images = convert_from_bytes(file_bytes, dpi=300)
            
            processed_images_bytes = []
            total_pages = len(pil_images)
            
            for i, pil_img in enumerate(pil_images):
                progress = int((i / total_pages) * 90)
                progress_bar.progress(progress)
                status_text.text(f"Processing Page {i+1}/{total_pages}...")
                
                open_cv_image = np.array(pil_img) 
                open_cv_image = open_cv_image[:, :, ::-1].copy() 

                # 1. Straighten
                deskewed = deskew_image(open_cv_image)

                # 2. HD Enhance
                enhanced = enhance_image(deskewed)

                # 3. Save
                img_pil_final = Image.fromarray(enhanced)
                img_byte_arr = io.BytesIO()
                
                # --- 关键修改：使用高质量保存 ---
                img_pil_final.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                processed_images_bytes.append(img_byte_arr.getvalue())

            status_text.text("Creating PDF...")
            final_pdf_bytes = img2pdf.convert(processed_images_bytes)
            
            progress_bar.progress(100)
            status_text.success("Done! Crystal clear.")
            
            st.download_button(
                label="📥 Download HD PDF",
                data=final_pdf_bytes,
                file_name=f"HD_{uploaded_file.name}",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error: {e}")
            if "poppler" in str(e).lower():
                st.warning("System Error: Poppler missing.")
