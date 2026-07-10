# MediExplain

**An Explainable AI-Based System for Multimodal Health Report Interpretation, Visual Diagnosis, and Specialist Recommendation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 Overview

MediExplain is an AI-powered medical report interpretation system that bridges the gap between complex medical data and patient comprehension. The platform supports:

- **Laboratory Report Analysis**: PDF upload with OCR, BioBERT entity recognition, and multi-disease risk prediction
- **Medical Imaging Analysis**: Chest X-ray and Brain MRI analysis with ResNet-50 and Grad-CAM explainability
- **Conversational AI Assistant**: Natural language Q&A about health reports
- **Explainable AI**: SHAP for ML models, Grad-CAM for vision models, RAG for clinical grounding

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- MongoDB (local or Atlas)
- 8GB+ RAM (for ML models)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd medexplain
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB URI, API keys, etc.
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Set up ML models** (See [Model Setup](#-ml-model-setup) below)

5. **Run with Docker**
   ```bash
   docker-compose up -d
   ```

6. **Access the application**
   - Web UI: http://localhost:8009
   - API Docs: http://localhost:8009/docs

---

## 🧠 ML Model Setup

### Required Models

The application requires the following trained models:

#### 1. Disease Prediction Models (`.pkl` files)
Place in `backend/ml/models/`:
- `kidney_disease_model.pkl`
- `thyroid_model.pkl`
- `breast_cancer_model.pkl`
- `stroke_model.pkl`
- `heart_disease_model.pkl`
- `liver_disease_model.pkl`
- `diabetes_model.pkl`

#### 2. Vision Models (`.pth` files)
Place in `backend/ml/models/`:
- `resnet50_chest_xray.pth` (NIH 14-disease classification)
- `resnet50_brain_mri.pth` (Tumor detection)

#### 3. BioBERT Model
Place in `backend/data/trained model data/`:
- `mediexplain_biobert_final/` (directory with model files)

### Training Models

Use the provided training scripts:

```bash
# Train disease prediction models
python backend/ml/colab_trainer.py

# Train vision models
python backend/ml/colab_vision_trainer.py
```

**Note**: Training requires GPU and can take several hours. Pre-trained models can be downloaded from [link to be added].

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │  HTML/CSS/JS + Jinja2
│   (Templates)   │
└────────┬────────┘
         │
┌────────▼────────┐
│   FastAPI       │  Python 3.11 + Uvicorn
│   Backend       │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │         │          │          │
┌───▼───┐ ┌──▼──┐  ┌────▼────┐ ┌──▼──┐
│MongoDB│ │Qdrant│ │ Ollama  │ │Groq │
│  DB   │ │Vector│ │  LLM    │ │ API │
└───────┘ └──────┘ └─────────┘ └─────┘
```

### Key Components

- **Backend**: FastAPI with modular architecture (8 modules)
- **ML Pipeline**: BioBERT → RAG → ML Models → LLM Synthesis
- **Vision Pipeline**: ResNet-50 → Grad-CAM → Vision LLM → RAG
- **Database**: MongoDB (primary), Qdrant (vector search)
- **LLM**: Llama 3 (local) + Groq API (cloud)

---

## 📊 Features

### Laboratory Analysis
- ✅ OCR text extraction from PDF reports
- ✅ BioBERT named entity recognition
- ✅ 6 disease prediction models (75-100% accuracy)
- ✅ SHAP explainability
- ✅ Historical trend analysis
- ✅ PDF report generation

### Medical Imaging
- ✅ Chest X-ray analysis (14 diseases, >95% accuracy)
- ✅ Brain MRI tumor detection
- ✅ Grad-CAM heatmaps
- ✅ Anatomical visualization
- ✅ Clinical urgency index

### AI Assistant
- ✅ Natural language Q&A
- ✅ RAG-grounded responses
- ✅ Medical literature citations

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=medexplain

# Qdrant
QDRANT_URL=http://localhost:6333

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b

# Groq API (optional)
GROQ_API_KEY=your_groq_api_key

# Security
JWT_SECRET_KEY=your_secret_key_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

---

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Check code coverage
pytest --cov=backend tests/
```

---

## 📦 Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Production Deployment

For production deployment (AWS, Render, etc.):
1. Set environment variables in your hosting platform
2. Ensure MongoDB and Qdrant are accessible
3. Upload ML models to persistent storage
4. Configure SSL/TLS certificates
5. Set up monitoring and logging

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Capstone Project - MSDS**

---

## 🙏 Acknowledgments

- NIH Chest X-Ray Dataset
- BioBERT Team
- FastAPI Framework
- Ollama Project

---

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

**Built with ❤️ using AI for Healthcare**
