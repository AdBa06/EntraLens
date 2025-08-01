# Lens Prototype

## Overview
Lens clusters and analyzes customer input for Security Copilot in two modes:
1. Mode 1: Intent Analysis  
2. Mode 2: Workload Grouping  

## Prerequisites
- Python 3.10+  
- Node.js 16+ and npm/yarn  

## Setup

### Backend
```bash
cd backend
cp .env.example .env
# Fill in your Azure OpenAI credentials in .env
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 to access the app.
