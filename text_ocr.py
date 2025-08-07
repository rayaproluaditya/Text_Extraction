
# ✅ Step 1: Install required libraries (only once per session)
!sudo apt install tesseract-ocr -y
!pip install pytesseract opencv-python pillow scikit-image matplotlib

# ✅ Step 2: Full Code
import cv2
import pytesseract
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Image as IPImage
import time
import os

# Tesseract path (already installed in Colab at this location)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ✅ Upload image
def upload_image():
    uploaded = files.upload()
    image_path = next(iter(uploaded.keys()))
    display(HTML("<h3>✅ Uploaded Image:</h3>"))
    display(IPImage(filename=image_path))
    return image_path

# ✅ Image Preprocessing (tuned for both printed & handwritten)
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding helps with handwritten text
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    return thresh

# ✅ Draw word boxes and collect predictions
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

# ✅ Basic heuristic: estimate content type
def estimate_content_type(image):
    # Heuristic: use contour roughness & pixel variability
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    stddev = np.std(gray)

    if stddev > 65:
        return "Both Printed and Handwritten"
    elif stddev > 35:
        return "Handwritten"
    else:
        return "Printed"

# ✅ Process full image
def process_image(image_path):
    image = cv2.imread(image_path)
    preprocessed = preprocess_image(image)

    # Tesseract config for word detection
    custom_config = r'--oem 3 --psm 6 -l eng'
    data = pytesseract.image_to_data(preprocessed, config=custom_config, output_type=pytesseract.Output.DICT)

    result_img, predictions = draw_word_boxes(image, data)

    # Estimate content type
    content_type = estimate_content_type(image)

    # Display final annotated image
    plt.figure(figsize=(16, 10))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("📌 Word Predictions with Boxes")
    plt.axis('off')
    plt.show()

    return predictions, content_type

# ✅ Save predictions to .txt file
def save_to_txt(predictions, content_type, filename='ocr_output.txt'):
    with open(filename, 'w') as f:
        f.write(f"📝 Detected Content Type: {content_type}\n\n")
        f.write("Word\t\tBoundingBox (x, y, w, h)\n")
        f.write("=" * 40 + "\n")
        for i, pred in enumerate(predictions):
            word = pred['word']
            x, y, w, h = pred['box']
            f.write(f"{word}\t\t({x}, {y}, {w}, {h})\n")
    return filename

# ✅ Master function
def main():
    display(HTML("<h2>🔼 Step 1: Upload your handwritten or printed image</h2>"))
    image_path = upload_image()

    display(HTML("<h2>🔍 Step 2: Processing image for word-level OCR...</h2>"))
    start = time.time()
    predictions, content_type = process_image(image_path)
    end = time.time()

    display(HTML(f"<h2>✅ Step 3: Word Predictions (Processed in {end - start:.2f} seconds)</h2>"))
    display(HTML(f"<h3>🧠 Detected Content Type: <span style='color:blue'>{content_type}</span></h3>"))

    display(HTML("<table border='1'><tr><th>Word #</th><th>Prediction</th><th>Box (x, y, w, h)</th></tr>"))
    for i, pred in enumerate(predictions):
        x, y, w, h = pred['box']
        display(HTML(f"<tr><td>{i+1}</td><td>{pred['word']}</td><td>({x},{y},{w},{h})</td></tr>"))
    display(HTML("</table>"))

    txt_file = save_to_txt(predictions, content_type)
    display(HTML(f"<h3>📄 OCR Output saved to <code>{txt_file}</code></h3>"))

    # Auto-download the .txt file
    files.download(txt_file)

# ✅ Run the app
main()
