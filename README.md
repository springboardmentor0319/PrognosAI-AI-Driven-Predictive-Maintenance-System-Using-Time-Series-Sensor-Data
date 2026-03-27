# PrognosAI: AI-Driven Predictive Maintenance System

## 🚀 Overview
PrognosAI is an advanced predictive maintenance solution designed to estimate the **Remaining Useful Life (RUL)** of industrial assets. By leveraging the **NASA C-MAPSS (Turbofan Engine Degradation Simulation) dataset**, the system identifies patterns in sensor data to predict failures before they occur.

## 🏗️ Architecture
This project is fully containerized using **Docker**, ensuring a seamless "plug-and-play" setup.
* **Backend:** FastAPI (Python) for high-performance data processing.
* **Database:** PostgreSQL for storing historical sensor readings.
* **Visualization:** Grafana dashboard for real-time RUL monitoring.
* **Deployment:** Orchestrated via Docker Compose.

## 📦 Getting Started

### Prerequisites
* **Docker Desktop** installed and running.
* **WSL 2** updated (for Windows users).

### Installation & Setup
1. **Prepare Environment:**
   - Ensure you have a file named `.env` in the root folder.
2. **Launch the System:**
   - Open your terminal in the project root and run:
   ```bash
   docker-compose up --build
