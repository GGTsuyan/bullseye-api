# Render Cloud Deployment Guide

## Overview
This API requires TensorFlow and the ML model files to function properly. It provides high-accuracy dart and dartboard detection using machine learning.

## Deployment Steps

### 1. Create a New Web Service on Render
- Go to [render.com](https://render.com)
- Click "New +" → "Web Service"
- Connect your GitHub repository

### 2. Configure the Service
- **Name**: `bullseye-api` (or your preferred name)
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn bullseye_api:app --host 0.0.0.0 --port $PORT`

### 3. Environment Variables
- **PORT**: Automatically set by Render
- **PYTHON_VERSION**: `3.9` (or higher)

### 4. Deploy
- Click "Create Web Service"
- Wait for the build to complete

## What Happens With the Model

### ✅ **Available Functionality:**
- **Dartboard Detection**: High-accuracy ML-based detection
- **Dart Detection**: Precise ML-based dart tip detection
- **Scoring**: Full scoring system with wedges, doubles, triples
- **API Endpoints**: All endpoints work normally
- **CORS**: Enabled for Flutter app connection

### 🚀 **Advantages:**
- **Detection Accuracy**: High accuracy using trained ML model
- **Performance**: Fast TensorFlow inference
- **Reliability**: Robust detection in various lighting conditions
- **Precision**: Accurate dart tip positioning for scoring

### 🔧 **ML Detection:**
- **Dartboard**: Uses trained model to detect dartboard boundaries
- **Darts**: ML-based object detection with confidence scores
- **Scoring**: Precise coordinate mapping for accurate scoring

## Testing the Deployment

### 1. Check Health
```bash
curl https://your-app-name.onrender.com/healthz
```

### 2. Check Root Endpoint
```bash
curl https://your-app-name.onrender.com/
```

### 3. Test with Flutter App
- Update the API URL in your Flutter app
- Test dartboard initialization
- Test dart detection

## Performance Notes

- **Cold Start**: ~10-15 seconds (normal for Render)
- **Response Time**: 200-500ms per request (CPU-based ML inference)
- **Memory Usage**: ~400-600MB (includes TensorFlow CPU model)
- **CPU Usage**: Moderate (TensorFlow CPU + OpenCV operations)
- **GPU**: Disabled (CPU-only for compatibility)

## Troubleshooting

### Common Issues:
1. **Build Fails**: Check Python version compatibility
2. **Import Errors**: Verify requirements.txt dependencies
3. **Port Issues**: Ensure `$PORT` environment variable is used
4. **CORS Errors**: Verify CORS middleware is enabled

### Logs:
- Check Render dashboard for build logs
- Monitor runtime logs for errors
- Verify all dependencies are installed

## Next Steps

### Option 1: Deploy with Model (Recommended)
- Full ML-based detection functionality
- High accuracy dart and dartboard detection
- Professional-grade performance

### Option 2: Optimize Model Size
- Consider model quantization for smaller size
- Use TensorFlow Lite for mobile deployment
- Optimize for Render's memory constraints

### Option 3: Use External ML Service
- Connect to Google Cloud Vision API
- Use AWS Rekognition
- Integrate with other ML services

## Support
- Check Render documentation for Python deployments
- Monitor service logs for specific errors
- Test endpoints individually to isolate issues
