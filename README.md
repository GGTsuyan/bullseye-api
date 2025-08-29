# Bullseye API - Dart Detection Service

A FastAPI service for detecting darts and scoring dartboard throws using computer vision and machine learning.

## 🚀 Quick Deploy to Render

### 1. Fork/Clone this repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Deploy to Render
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: `bullseye-api`
   - **Environment**: `Docker`
   - **Branch**: `main`
   - **Root Directory**: Leave empty
6. Click "Create Web Service"

### 3. Add Model Files
After deployment, you'll need to add your TensorFlow model files to the `models/saved_model/` directory.

## 🔧 Local Development

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run locally
```bash
python bullseye_api.py
```

### Test endpoints
- `POST /init-board` - Initialize dartboard
- `POST /detect-dart` - Detect darts
- `POST /reset-turn` - Reset turn
- `GET /debug-visual` - Get debug visualization

## 📁 Project Structure
```
├── bullseye_api.py      # Main FastAPI application
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── render.yaml          # Render deployment config
└── models/              # TensorFlow model directory
    └── saved_model/     # Your trained model files
```

## 🌐 API Endpoints

### Initialize Board
```bash
curl -X POST "https://your-app.onrender.com/init-board" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dartboard_image.jpg"
```

### Detect Dart
```bash
curl -X POST "https://your-app.onrender.com/detect-dart" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dart_image.jpg"
```

## 📝 Notes
- Free tier sleeps after 15 minutes of inactivity
- First request after sleep may take 1-2 minutes
- Model files need to be uploaded separately
