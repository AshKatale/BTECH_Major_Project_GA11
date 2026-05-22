# 🔬 DermaAI : Federated Learning Skin Cancer Prediction System

> A privacy-preserving, multi-institutional skin cancer detection platform powered by **Federated Learning**, **EfficientNet-B0**, and the **Flower framework** — enabling collaborative AI training without sharing patient data.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Environment Configuration](#-environment-configuration)
- [Running the System](#-running-the-system)
- [Federated Learning Workflow](#-federated-learning-workflow)
- [API Endpoints](#-api-endpoints)
- [Skin Cancer Classes](#-skin-cancer-classes)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Demonstration Guide](#-demonstration-guide)
- [Troubleshooting](#-troubleshooting)

---

## 🧠 Overview

This B.Tech Major Project implements a **Federated Learning (FL) system** for skin cancer detection that allows multiple hospitals or clinics to collaboratively train a deep learning model **without sharing raw patient data**. Each institution trains locally on its own dataset, sharing only model weight updates — ensuring **GDPR/HIPAA compliance** and patient privacy.

### Key Features

| Feature | Description |
|---|---|
| 🔒 **Privacy-Preserving** | Patient images never leave local nodes |
| 🤝 **Collaborative Learning** | Multiple hospitals improve a shared model |
| 🧬 **EfficientNet-B0** | Pre-trained ImageNet model, 7-class skin lesion classification |
| 🌐 **Full-Stack** | React frontend + Express backend + Flask ML APIs |
| 🖥️ **Desktop App** | Electron app for hospital-side FL client management |
| 📊 **Real-Time Monitoring** | Live FL round tracking, accuracy metrics, prediction history |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────┐
│             WEB DASHBOARD (React/Vite)               │
│                  localhost:5173                       │
└──────────────────────┬───────────────────────────────┘
                       │ HTTP/REST (JWT Auth)
┌──────────────────────▼───────────────────────────────┐
│            EXPRESS BACKEND SERVER :3001               │
│   Auth (JWT) │ Predictions │ FL Management            │
└───┬──────────────────┬──────────────────┬────────────┘
    │                  │                  │
┌───▼───┐      ┌───────▼────┐    ┌───────▼──────┐
│MongoDB│      │ PostgreSQL │    │  Flask ML API│
│(preds)│      │(users/logs)│    │  :5000/:6000 │
└───────┘      └────────────┘    └──────────────┘

┌──────────────────────────────────────────────────────┐
│       FEDERATED LEARNING SERVER (Flower) :8080        │
│  FedAvg Strategy → Aggregates model weights           │
│         ↕ gRPC (bi-directional)                       │
│  [Hospital 1] [Hospital 2] [Hospital 3] ... [N]       │
│  Each trains locally → sends weights only             │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│         DESKTOP APP (Electron)                        │
│  Manages local FL client training per hospital        │
│  Connects to FL Server & Express Backend              │
└──────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Major_Project_Github/
│
├── client/                     # React + Vite Frontend
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/              # Route-level page components
│   │   ├── context/            # React context (auth, state)
│   │   ├── services/           # API service calls (axios)
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── public/
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
│
├── server/                     # Node.js Express Backend
│   ├── config/                 # DB config (MongoDB, PostgreSQL)
│   ├── controllers/            # Route handlers
│   ├── middleware/             # JWT auth, file upload, rate limiting
│   ├── models/                 # Mongoose & Sequelize models
│   ├── routes/                 # API route definitions
│   ├── scripts/                # DB init scripts
│   ├── uploads/                # Uploaded image storage
│   ├── server.js               # Entry point
│   ├── .env.example
│   └── package.json
│
├── ml-model/                   # Flask ML Inference API
│   ├── models/                 # Saved PyTorch model weights (.pt)
│   ├── app.py                  # Flask server (port 5000)
│   ├── skin_cancer_model.py    # EfficientNet-B0 model definition
│   ├── train_model.py          # Standalone training script
│   ├── aggregate_models.py     # Model aggregation utilities
│   └── requirements.txt
│
├── fl-server/                  # Flower Federated Learning Server
│   ├── models/                 # Global model weights per FL round
│   ├── client_updates/         # Temporary client weight updates
│   ├── app.py                  # Flower FL server (port 8080 gRPC, 6000 HTTP)
│   ├── aggregator.py           # FedAvg aggregation logic
│   ├── model.py                # Shared model architecture
│   ├── round_manager.py        # FL round orchestration
│   ├── Dockerfile
│   ├── .env.example
│   └── requirements.txt
│
├── desktop-app/                # Electron Desktop App (Hospital Client)
│   ├── fl_client/              # Python FL client scripts
│   │   ├── client.py           # Flower NumPy client
│   │   ├── trainer.py          # Local training logic
│   │   ├── model.py            # Model definition
│   │   ├── training_runner.py  # Training orchestration
│   │   ├── inference_runner.py # Local inference
│   │   ├── evaluate_model.py   # Model evaluation
│   │   ├── gemini_analyzer.py  # AI-assisted analysis
│   │   ├── scheduler.py        # Training scheduler
│   │   └── requirements.txt
│   ├── assets/
│   ├── main.js                 # Electron main process
│   ├── preload.js              # Electron preload scripts
│   └── package.json
│
├── SYSTEM_DOCUMENTATION.md     # Full technical documentation
├── ARCHITECTURE_PROMPT.md      # Architecture design notes
├── .gitignore
└── README.md                   # ← You are here
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Frontend** | React + Vite + TailwindCSS | React 19, Vite 8 |
| **Backend** | Node.js + Express | v18+ |
| **ML Inference** | Flask + PyTorch | Flask 2+, PyTorch 2+ |
| **FL Framework** | Flower (`flwr`) | Latest |
| **ML Model** | EfficientNet-B0 (timm) | Pre-trained ImageNet |
| **Databases** | MongoDB + PostgreSQL | Latest |
| **ODM/ORM** | Mongoose + Sequelize | Latest |
| **Auth** | JWT (jsonwebtoken) + bcryptjs | — |
| **Desktop** | Electron | v27+ |
| **Containerization** | Docker (FL Server) | — |

---

## ✅ Prerequisites

Ensure the following are installed before proceeding:

- **Node.js** v18+ — [Download](https://nodejs.org/)
- **Python** 3.9–3.11 — [Download](https://python.org/)
- **MongoDB** (local or Atlas) — [Download](https://mongodb.com/)
- **PostgreSQL** — [Download](https://postgresql.org/)
- **Git** — [Download](https://git-scm.com/)
- *(Optional)* **CUDA 12.1** compatible GPU for accelerated training

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AshKatale/-BTECH_Major_Project_GA11.git
cd -BTECH_Major_Project_GA11
```

### 2. Install Client (Frontend)

```bash
cd client
npm install
```

### 3. Install Server (Backend)

```bash
cd ../server
npm install
```

### 4. Install Desktop App

```bash
cd ../desktop-app
npm install
```

### 5. Install ML Model Python Dependencies

```bash
cd ../ml-model
pip install -r requirements.txt
# For GPU/CUDA 12.1 support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 6. Install FL Server Python Dependencies

```bash
cd ../fl-server
pip install -r requirements.txt
```

### 7. Install Desktop FL Client Python Dependencies

```bash
cd ../desktop-app/fl_client
pip install -r requirements.txt
```

---

## 🔧 Environment Configuration

### Server (`server/.env`)

Copy the example and fill in your values:

```bash
cp server/.env.example server/.env
```

```env
PORT=3001
NODE_ENV=development

# MongoDB
MONGO_URI=mongodb://localhost:27017/skin-cancer-db

# JWT
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRE=7d

# ML & FL API URLs
ML_API=http://localhost:5000
FL_API=http://localhost:6000

# File Upload
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=52428800

# CORS
CORS_ORIGIN=http://localhost:5173,http://localhost:3000
```

### FL Server (`fl-server/.env`)

```bash
cp fl-server/.env.example fl-server/.env
```

```env
FL_SERVER_PORT=6000
ROUND_DURATION_SECONDS=86400
MODELS_DIR=./models/global
CLIENT_UPDATES_DIR=./client_updates
CORS_ORIGINS=http://localhost:5173,http://localhost:3001
```

---

## ▶️ Running the System

Run each component in a **separate terminal**:

### Terminal 1 — Express Backend

```bash
cd server
npm run dev
# Server starts on http://localhost:3001
```

### Terminal 2 — React Frontend

```bash
cd client
npm run dev
# Frontend starts on http://localhost:5173
```

### Terminal 3 — ML Inference API (Flask)

```bash
cd ml-model
python app.py
# Flask ML server starts on http://localhost:5000
```

### Terminal 4 — Federated Learning Server

```bash
cd fl-server
python app.py
# FL HTTP server: http://localhost:6000
# FL gRPC server: localhost:8080
```

### Terminal 5 — Desktop App (Electron) *(Optional)*

```bash
cd desktop-app
npm run dev
# Opens the Electron desktop application
```

### Initialize Databases *(First Time Only)*

```bash
cd server
npm run init-db
```

---

## 🌐 Access Points

| Service | URL |
|---|---|
| Web Dashboard | http://localhost:5173 |
| Express API | http://localhost:3001 |
| ML Inference API | http://localhost:5000 |
| FL HTTP API | http://localhost:6000 |
| FL gRPC Server | localhost:8080 |

---

## 🔄 Federated Learning Workflow

### Starting an FL Training Round

1. **Open the Web Dashboard** → navigate to the **Federated Learning** section
2. **Start FL Server**: Click "Start FL Server" or run `python fl-server/app.py`
3. **Connect Clients**: Each hospital launches the **Desktop App** and connects to the FL server
4. **Initiate Training Round**: Trigger from the dashboard or API
5. **Monitor Progress**: Watch real-time round metrics on the dashboard
6. **Inference**: Use the trained global model for predictions via the web UI

### FL Round Process (FedAvg)

```
Round N:
  ├─► FL Server distributes current global model to all clients
  ├─► Each hospital trains locally on private data (1 epoch)
  ├─► Clients send model weight updates back (no raw data)
  ├─► Server aggregates using FedAvg:
  │       global = Σ (n_i / Σn) × weights_i
  └─► Aggregated global model saved → distributed for Round N+1
```

### Running a Manual FL Client

```bash
cd desktop-app/fl_client
python client.py <client_id> <server_address> [dataset_path]

# Example:
python client.py 1 127.0.0.1:8080 "D:\Skin Cancer Dataset"
```

---

## 📡 API Endpoints

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/login` | User login (returns JWT) |
| POST | `/api/auth/signup` | User registration |
| POST | `/api/auth/logout` | Logout |
| GET | `/api/auth/profile` | Get user profile |

### Predictions

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predictions/predict` | Single image prediction (standard model) |
| POST | `/api/predictions/fl/predict` | Single image prediction (FL trained model) |
| POST | `/api/predictions/batch` | Batch image predictions |
| GET | `/api/predictions/history` | User prediction history |
| GET | `/api/predictions/stats` | User statistics |
| GET | `/api/predictions/fl/info` | FL model metadata |

### Federated Learning

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/federated-learning/server/start` | Start FL server |
| POST | `/api/federated-learning/client/start` | Start FL client |
| POST | `/api/federated-learning/rounds/initiate` | Initiate training round |
| GET | `/api/federated-learning/analytics` | FL training analytics |
| GET | `/api/federated-learning/:trainingId/status` | Training status |

### ML Inference (Flask, port 5000)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/predict` | Single image prediction |
| GET | `/api/fl-model-info` | Model info |

---

## 🔬 Skin Cancer Classes

The model classifies skin lesions into **7 categories** (HAM10000 dataset):

| ID | Code | Class Name | Risk |
|---|---|---|---|
| 0 | `akiec` | Actinic Keratosis | Medium |
| 1 | `bcc` | Basal Cell Carcinoma | High |
| 2 | `bkl` | Benign Keratosis | Low |
| 3 | `df` | Dermatofibroma | Low |
| 4 | `mel` | Melanoma | Very High |
| 5 | `nv` | Melanocytic Nevus | Low |
| 6 | `vasc` | Vascular Lesions | Medium |

---

## 📈 Model Performance

| FL Round | Training Loss | Accuracy |
|---|---|---|
| Round 1 | 3.08 | 27.5% |
| Round 2 | 0.12 | 96.2% |
| Round 3 | 0.12 | 97.5% |
| **Round 4** | **0.07** | **98.8% ✓ Best** |
| Round 5 | 0.21 | 93.8% |

- **Architecture**: EfficientNet-B0 (~4.2M parameters)
- **Input**: 224×224×3 images
- **Dataset**: HAM10000 (10,015 dermoscopy images)
- **Optimizer**: Adam (LR=0.001), ReduceLROnPlateau scheduler
- **Loss**: CrossEntropyLoss

---

## 🚢 Deployment

### FL Server with Docker

```bash
cd fl-server
docker build -t fl-server .
docker run -p 6000:6000 -p 8080:8080 --env-file .env fl-server
```

### Client (Frontend) Build

```bash
cd client
npm run build
# Output in client/dist/ — serve with any static host (Vercel, Netlify, etc.)
```

### Desktop App Build

```bash
cd desktop-app
# Windows installer:
npm run build-win

# macOS:
npm run build-mac

# Linux:
npm run build-linux

# Output in desktop-app/release/
```

### Render / Cloud Deployment

Update the following in `fl-server/.env`:

```env
CORS_ORIGINS=https://your-frontend-domain.com,https://your-backend-domain.com
```

---

## 🎥 Demonstration Guide

### Step-by-Step Demo Flow

1. **Start all services** (see [Running the System](#️-running-the-system))
2. **Open** http://localhost:5173 in browser
3. **Register / Login** as a doctor or admin
4. **Upload a skin lesion image** → receive instant prediction with confidence score and risk level
5. **View Prediction History** → filter by date, risk level, class
6. **Navigate to Federated Learning Dashboard**:
   - Start the FL Server
   - Connect a hospital client via the Desktop App
   - Trigger a training round
   - Monitor round progress and aggregated accuracy in real-time
7. **Switch Inference Model** → compare standard vs FL-trained model predictions
8. **Batch Prediction** → upload multiple images, export results as CSV

### Key Demo Points

- Show the privacy guarantee: client never uploads raw images to FL server
- Demonstrate accuracy improvement across FL rounds (Round 1 → Round 4)
- Show role-based access control (doctor vs admin views)

---

## 🐛 Troubleshooting

### MongoDB Connection Error

```bash
# Check MongoDB is running
mongod --dbpath /data/db
# Or start MongoDB service
net start MongoDB
```

### Python Module Not Found

```bash
# Ensure you're in the correct virtualenv and installed requirements
pip install -r requirements.txt
```

### Port Already in Use

```bash
# Windows — find and kill process on a port
netstat -ano | findstr :3001
taskkill /PID <PID> /F
```

### CUDA / GPU Not Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, install CUDA toolkit and PyTorch CUDA build:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### FL Client Cannot Connect to Server

- Verify FL server is running on `localhost:8080`
- Check firewall rules allow gRPC port 8080
- Ensure correct `server_address` passed to client script

---

## 📄 License

This project is developed as a **B.Tech Major Project** for academic purposes.

---

## 👨‍💻 Author

**Ash Katale** — [GitHub: AshKatale](https://github.com/AshKatale)

---

> 📖 For complete technical documentation, see [SYSTEM_DOCUMENTATION.md](./SYSTEM_DOCUMENTATION.md)
