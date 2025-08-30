FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV (minimal set)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Create models directory
RUN mkdir -p models/saved_model

# Set memory optimization environment variables
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV TF_MEMORY_ALLOCATION=0.5
ENV TF_CPP_VMODULE=tensorflow=0
ENV OMP_NUM_THREADS=1

# Expose port (Render will override this)
EXPOSE 8000

# Default command (will be overridden by render.yaml)
CMD ["uvicorn", "bullseye_api:app", "--host", "0.0.0.0", "--port", "8000"]
