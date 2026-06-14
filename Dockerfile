FROM python:3.11-slim

WORKDIR /app

# System deps needed by mediapipe / opencv
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces requires the app to listen on port 7860
EXPOSE 7860

CMD ["python", "app.py"]