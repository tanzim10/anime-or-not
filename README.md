# AnimeOrNot (AoN) 🎨

AnimeOrNot (AoN) is a **FastAPI-powered image classification service** that leverages deep learning to distinguish between anime and cartoon images. Despite their visual similarities, this API exploits subtle structural differences to provide accurate classification with prediction probabilities.

## 🎯 Project Overview

While anime and cartoons are both forms of animation with similar visual styles, there are distinct structural differences that can be exploited for classification. This project implements a **ResNet50-based deep learning model** trained to identify these subtle differences and classify images accordingly.

### Key Features
- 🚀 **FastAPI REST API** with automatic documentation
- 🧠 **ResNet50 deep learning model** for accurate classification
- 📊 **Prediction probabilities** for confidence scoring
- 🐳 **Docker containerization** for easy deployment
- 📈 **Health monitoring** and model information endpoints

## 🛠️ Technology Stack

- **Backend:** Python 3.10.16, FastAPI, Uvicorn
- **ML Framework:** PyTorch, TorchVision
- **Image Processing:** Pillow (PIL)
- **Containerization:** Docker, Docker Compose
- **Documentation:** Pydantic, OpenAPI/Swagger
- **Development:** Git, YAML configuration

## 📋 Prerequisites

- **Python 3.10+** or **Docker**
- **Git** for version control
- **Model file:** `best_model.pth` (see [Model Setup](#model-setup))

## 🚀 Quick Start

### Option 1: Docker (Recommended)

#### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed and running
- [Docker Compose](https://docs.docker.com/compose/install/) (optional)

#### Build and Run
```bash
# Clone the repository
git clone <your-repo-url>
cd AnimeOrNot

# Build the Docker image
docker build -t anime-or-not:latest .

# Run the container
docker run -dp 8000:8000 anime-or-not:latest
```

#### Using Docker Compose
```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Local Development

#### Setup Virtual Environment
```bash
# Create virtual environment
python -m venv .aenv

# Activate virtual environment
# On macOS/Linux:
source .aenv/bin/activate
# On Windows:
.aenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Run Locally
```bash
# Start the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or with custom settings
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## 📡 API Endpoints

### Base URL
```
http://localhost:8000
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | **Interactive API documentation** (Swagger UI) |
| `/redoc` | GET | Alternative API documentation |
| `/model/health` | GET | **Health check** - API status |
| `/model/model-info` | GET | **Model information** - architecture, size, parameters |
| `/model/predict` | POST | **Prediction endpoint** - classify anime vs cartoon |

### API Usage Examples

#### Health Check
```bash
curl http://localhost:8000/model/health
```
**Response:**
```json
{
  "status": "API is ready for serving model predictions",
  "code": 200
}
```

## 🧠 Model Setup

### Getting the Model File

The `best_model.pth` file is **not included** in this repository due to its large size. Here are the setup options:

#### Option 1: Download from Cloud Storage
```bash
# Download the model file
curl -L "https://your-storage-url.com/best_model.pth" -o best_model.pth
```

#### Option 2: Place in Project Root
1. Download the model file from your storage location
2. Place `best_model.pth` in the project root directory
3. The Docker build will automatically copy it

#### Option 3: Use Environment Variable
```bash
# Set model URL in environment
export MODEL_URL="https://your-storage-url.com/best_model.pth"

# Build Docker with model URL
docker build --build-arg MODEL_URL=$MODEL_URL .
```

### Model Specifications
- **Architecture:** ResNet50
- **Classes:** 2 (Anime, Cartoon)
- **Input Size:** 224x224 pixels
- **Format:** PyTorch (.pth)
- **Preprocessing:** ImageNet normalization

## 🏗️ Project Structure

```
AnimeOrNot/
├── app/                    # FastAPI application
│   ├── main.py            # Application entry point
│   ├── routers/           # API route definitions
│   │   └── modelRouter.py # Model endpoints
│   ├── services/          # Business logic
│   │   └── modelService.py # Model service
│   ├── schemas/           # Pydantic models
│   │   └── modelSchema.py # API schemas
│   ├── exceptions/        # Custom exceptions
│   └── config/           # Configuration
├── src/                   # Source code
│   ├── model/            # ML model code
│   │   ├── models.py     # Model architecture
│   │   └── prediction.py # Prediction functions
│   └── utils/            # Utility functions
├── data/                 # Data directory (ignored)
├── tests/               # Test files
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── requirements.txt     # Python dependencies
├── config.yaml         # Application configuration
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🔧 Configuration

The application uses `config.yaml` for configuration management:

```yaml
app:
  name: "AnimeOrNot API"
  version: "1.0.0"
  host: "0.0.0.0"
  port: 8000

model:
  name: "ResNet50"
  num_classes: 2
  input_size: [224, 224]
  class_names: ["Anime", "Cartoon"]
```

## 🧪 Testing

### Manual Testing
1. **Start the server** (Docker or local)
2. **Open Swagger UI:** http://localhost:8000/docs
3. **Test endpoints** using the interactive interface

### Health Check
```bash
curl http://localhost:8000/model/health
```

### Model Info
```bash
curl http://localhost:8000/model/model-info
```

### Prediction Test
```bash
# Using curl
curl -X POST "http://localhost:8000/model/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"

# Using Python requests
import requests

with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/model/predict',
        files={'file': f}
    )
print(response.json())
```

## 🐳 Docker Commands

### Build Image
```bash
docker build -t anime-or-not:latest .
```

### Run Container
```bash
# Basic run
docker run -dp 8000:8000 anime-or-not:latest

# With custom name
docker run -dp 8000:8000 --name aon-container anime-or-not:latest

# With volume mount (for development)
docker run -dp 8000:8000 -v $(pwd)/app:/app/app anime-or-not:latest
```


### Common Issues

#### 1. Model File Missing
**Error:** `FileNotFoundError: best_model.pth`
**Solution:** Download the model file and place it in the project root

#### 2. Port Already in Use
**Error:** `Address already in use`
**Solution:** 
```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill -9 <PID>
```

#### 3. Docker Build Fails
**Error:** `failed to compute cache key`
**Solution:** 
```bash
# Clean Docker cache
docker system prune -a
# Rebuild
docker build --no-cache -t anime-or-not:latest .
```



## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** for the excellent web framework
- **PyTorch** for the deep learning capabilities
- **Docker** for containerization

## 📞 Support

For support, please open an issue on GitHub or contact the maintainers.

---
