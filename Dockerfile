


FROM python:3.9-slim

# Install Tesseract and dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    # For image processing:
    ffmpeg \
    libsm6 \
    libxext6 \
    # For matplotlib (if needed):
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Set Tesseract path explicitly
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
