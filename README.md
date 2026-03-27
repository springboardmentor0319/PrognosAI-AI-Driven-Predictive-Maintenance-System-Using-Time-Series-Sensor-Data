# PrognosAI — AI-Driven Predictive Maintenance System

## 📌 Overview

PrognosAI is a machine learning-based system that predicts the **Remaining Useful Life (RUL)** of turbofan engines using time-series sensor data.
It helps in identifying potential failures early and supports efficient maintenance planning.

---

## 🚀 Tech Stack

* Python
* XGBoost
* FastAPI
* PostgreSQL
* Grafana
* Docker

---

## ⚙️ Features

* Predicts engine failure using ML models
* REST API using FastAPI
* Visualization using Grafana dashboards
* Fully containerized using Docker
* Easy deployment and scalability

---

## 🐳 Running the Project Using Docker

### Step 1: Clone the Repository

```bash
git clone https://github.com/springboardmentor0319/PrognosAI-AI-Driven-Predictive-Maintenance-System-Using-Time-Series-Sensor-Data.git
cd PrognosAI-AI-Driven-Predictive-Maintenance-System-Using-Time-Series-Sensor-Data
```

---

### Step 2: Build and Start Containers

```bash
docker-compose up --build
```

---

### Step 3: Access Services

* FastAPI → http://localhost:8000
* Swagger UI → http://localhost:8000/docs
* Grafana → http://localhost:3000

---

### Step 4: Stop Containers

```bash
docker-compose down
```

---

## 📡 API Testing

Use the following sample input to test the prediction API:

```json
{
  "dataset": "FD002",
  "engine_ids": [1, 2, 3, 4, 5],
  "max_engines": 5
}
```

You can test using:

* Swagger UI (browser)
* Postman

---

## 📊 Grafana Dashboard

* Open: http://localhost:3000
* Username: admin
* Password: admin

Import dashboards from the `grafana_dashboard` folder.

---

## 📁 Project Structure

```
├── docker/
├── docs/
├── grafana_dashboard/
├── scripts/
├── ui/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
```

---

## 🎯 Applications

* Aircraft engine maintenance
* Predictive maintenance in industries
* Industrial IoT systems

---

## 📌 Conclusion

This project combines machine learning and Docker-based deployment to build a scalable and real-time predictive maintenance solution.

