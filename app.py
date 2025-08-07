# ✅ Required Libraries
import cv2
import pytesseract
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os
import matplotlib.pyplot as plt

# ✅ Tesseract Path for Linux (modify for Windows if needed)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ✅ Title and Page Config
st.set_page_config(page_title="🧠 OCR Analyzer", layout="centered", page_icon="📄")

# ✅ Header UI
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color:#4CAF50;'>🧠 Smart OCR: Handwritten vs Printed Detection</h1>
        <h4>Upload any handwritten or printed image and extract words with bounding boxes</h4>
    </div>
""", unsafe_allow_html=True)

# ✅ Upload Image
uploaded_file = st.file_uploader("📤 Upload your image (PNG, JPG)", type=["png", "jpg", "jpeg"])

# ✅ Preprocessing Function
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    return thresh

# ✅ Heuristic Content Type Estimator
def estimate_content_type(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stddev = np.std(gray)
    if stddev > 65:
        return "Both Printed and Handwritten"
    elif stddev > 35:
        return "Handwritten"
    else:
        return "Printed"

# ✅ Draw Word Boxes
def draw_word_boxes(image, ocr_data):
    output = image.copy()
    predictions = []
    n_boxes = len(ocr_data['text'])
    for i in range(n_boxes):
        word = ocr_data['text'][i].strip()
        if word:
            (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
            predictions.append({'word': word, 'box': (x, y, w, h)})
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, word, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return output, predictions

# ✅ Save Results to TXT
def save_to_txt(predictions, content_type):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", dir=".") as f:
        f.write(f"Detected Content Type: {content_type}\n\n")
        f.write("Word\t\tBoundingBox (x, y, w, h)\n")
        f.write("=" * 40 + "\n")
        for pred in predictions:
            word = pred['word']
            x, y, w, h = pred['box']
            f.write(f"{word}\t\t({x}, {y}, {w}, {h})\n")
        return f.name

# ✅ Main Logic
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Processing Image..."):
        preprocessed = preprocess_image(image)
        custom_config = r'--oem 3 --psm 6 -l eng'
        data = pytesseract.image_to_data(preprocessed, config=custom_config, output_type=pytesseract.Output.DICT)
        result_img, predictions = draw_word_boxes(image, data)
        content_type = estimate_content_type(image)

    # Show Results
    st.success(f"✅ Detected Content Type: **{content_type}**")
    st.markdown("### 🔤 Detected Words with Bounding Boxes")
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.markdown("### 📋 Word Predictions")
    for i, pred in enumerate(predictions):
        word = pred['word']
        x, y, w, h = pred['box']
        st.write(f"**{i+1}.** `{word}` at `(x={x}, y={y}, w={w}, h={h})`")

    # Download Button
    txt_path = save_to_txt(predictions, content_type)
    with open(txt_path, "rb") as file:
        st.download_button(label="📥 Download OCR Result (.txt)", data=file, file_name="ocr_output.txt", mime="text/plain")

    # Cleanup
    os.remove(txt_path)

# ✅ Footer
st.markdown("""
<hr>
<div style='text-align: center; font-size: 0.9em; color: grey;'>
    Developed with ❤️ using <b>Streamlit</b> & <b>Tesseract OCR</b> | 2025
</div>
""", unsafe_allow_html=True)
