FROM python:3.9-slim

# 1. Install Tesseract with ALL dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \  # English language pack
    tesseract-ocr-all \  # All language packs (optional)
    libtesseract-dev \
    libleptonica-dev \
    # Image processing dependencies:
    ffmpeg \
    libsm6 \
    libxext6 \
    # Clean up to reduce image size:
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Install Python dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the app
COPY . .

# 5. Verify Tesseract installation (debugging)
RUN tesseract --version

# 6. Explicitly set environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV PATH="/usr/bin/tesseract:${PATH}"

# 7. Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
