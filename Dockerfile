# Base image (Python 3.9 slim for smaller size)
FROM python:3.9-slim

# 1. Install system dependencies (Tesseract + OpenCV requirements)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first (for layer caching)
COPY requirements.txt ./

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application
COPY . .

# 6. Explicitly set Tesseract path (optional but recommended)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# 7. Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
