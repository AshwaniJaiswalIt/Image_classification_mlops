# Docker API Setup - Step-by-Step Guide

## ✅ Files Created Successfully!

Your API is ready in the `api/` folder:

```
api/
├── app.py                     ✅ FastAPI application
├── Dockerfile                 ✅ Container definition
├── requirements.txt           ✅ Dependencies
├── .dockerignore             ✅ Build optimization
├── test_api.py               ✅ Test suite
├── README.md                 ✅ Documentation
└── models/
    ├── random_forest_model.pkl ✅ Trained model
    └── imputer.pkl            ✅ Preprocessor
```

---

## 🚀 Next Steps

### Step 1: Install Docker Desktop

1. **Download Docker Desktop:**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"
   - Run the installer

2. **After Installation:**
   - Restart your computer
   - Launch "Docker Desktop" from Start menu
   - Wait for the whale icon to appear in system tray
   - Accept terms and complete setup

3. **Verify Installation:**
   ```powershell
   docker --version
   # Should show: Docker version 24.x.x
   
   docker run hello-world
   # Should download and run test container
   ```

---

### Step 2: Build Docker Image

Once Docker is installed:

```powershell
# Navigate to api folder
cd api

# Build the image (takes ~3-5 minutes first time)
docker build -t heart-disease-api .
```

**What happens:**
- Downloads Python base image (~150MB)
- Installs dependencies
- Copies your code and models
- Creates final image (~500MB)

---

### Step 3: Run Docker Container

```powershell
# Run container in background
docker run -d -p 8000:8000 --name heart-api heart-disease-api

# Check if container is running
docker ps
```

**Expected output:**
```
CONTAINER ID   IMAGE                 STATUS         PORTS
abc123...      heart-disease-api     Up 5 seconds   0.0.0.0:8000->8000/tcp
```

---

### Step 4: Test the API

**Option A: Open browser**
```
http://localhost:8000/docs
```
- You'll see interactive Swagger UI
- Click `/predict` → "Try it out"
- Use the example JSON
- Click "Execute"

**Option B: Run test script**
```powershell
# Make sure you're in the api folder
python test_api.py
```

**Option C: Manual curl test**
```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

---

### Step 5: Stop and Clean Up

```powershell
# Stop container
docker stop heart-api

# Remove container
docker rm heart-api

# (Optional) Remove image
docker rmi heart-disease-api
```

---

## 📋 Quick Command Reference

### Docker Commands
```powershell
# Build
docker build -t heart-disease-api .

# Run (foreground - see logs)
docker run -p 8000:8000 heart-disease-api

# Run (background)
docker run -d -p 8000:8000 --name heart-api heart-disease-api

# List running containers
docker ps

# View logs
docker logs heart-api

# Stop
docker stop heart-api

# Remove
docker rm heart-api
```

### API Endpoints
```
http://localhost:8000/         → API info
http://localhost:8000/health   → Health check
http://localhost:8000/docs     → Interactive docs
http://localhost:8000/predict  → Make prediction (POST)
```

---

## 🎯 What to Submit for Assignment

1. **Screenshot of Docker running:**
   ```powershell
   docker ps
   ```

2. **Screenshot of API docs:**
   - Browser: `http://localhost:8000/docs`

3. **Screenshot of successful prediction:**
   - Test output or browser result

4. **Code files:**
   - `app.py`
   - `Dockerfile`
   - `requirements.txt`

---

## 🐛 Troubleshooting

### Docker not found
- Install Docker Desktop
- Restart terminal after installation
- Make sure Docker Desktop is running

### Port 8000 already in use
```powershell
# Use different port
docker run -p 8001:8000 heart-disease-api
# Then access at http://localhost:8001
```

### Container won't start
```powershell
# Check logs
docker logs heart-api

# Run in foreground to see errors
docker run -p 8000:8000 heart-disease-api
```

### Model not loading
```powershell
# Verify model files are in the image
docker run heart-disease-api ls -la models/
```

---

## ✅ Summary

**What we built:**
- ✅ FastAPI application with `/predict` endpoint
- ✅ Docker container for deployment
- ✅ Input validation (Pydantic)
- ✅ Comprehensive test suite
- ✅ Interactive API documentation
- ✅ Health checks and monitoring

**Next steps:**
1. Install Docker Desktop
2. Build the image: `docker build -t heart-disease-api .`
3. Run the container: `docker run -p 8000:8000 heart-disease-api`
4. Test: `http://localhost:8000/docs`
5. Take screenshots for assignment

**Need help?** Check `api/README.md` for detailed documentation.
