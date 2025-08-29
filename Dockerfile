FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libglu1-mesa \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Create models directory (you'll need to upload your model files)
RUN mkdir -p models/saved_model

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "bullseye_api:app", "--host", "0.0.0.0", "--port", "8000"]
