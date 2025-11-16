# Ripe & Unripe Fruit Classification MLOps Project

This repository implements a complete MLOps workflow for a fruit classification model using ResNet50V2, MLflow, CI/CD with GitHub Actions, Docker deployment, and drift monitoring.

### Project Overview

This project demonstrates a full MLOps pipeline:

### Model Training & Logging

Conducted training experiments with different optimizers and hyperparameters with Grid search, Random Search and Bayesian optimization techniques.

All runs are logged to MLflow, including parameters, metrics, and artifacts.

### Model Selection & Registry

The best-performing model is selected from all experiments.

Registered in MLflow Model Registry and promoted through Staging → Production.

### Version Control

Code, dataset, and model are versioned using Git and DVC for reproducibility.

### CI/CD Pipeline (Deployment + Local Monitoring)

GitHub Actions automates:

1. Pulling versioned code and DVC-tracked data.

2. Installing dependencies.

3. Running a demo/reproducible training script (CICD script).

4. Building a Docker image containing the environment and Production model.

5. Deploying the Flask API container serving the Production model.

6. Running local drift monitoring/logging for metrics, confidence, and class distribution.

### Production API (Dockerized)

The model is served via a Dockerized Flask API, ensuring environment consistency and portability.

Accessible locally after deployment.

### Monitoring & Drift Detection

Monitoring for model performance, top-class confidence, and class distribution.

Alerts can be triggered if drift or degradation is detected.

Logs are visible locally on mlflow ui.

### Setup & Usage

1. Clone the Repository

git clone https://github.com/ssn-nishshanka/Fruit_Classification_MLOps.git

cd Fruit_Classification_ML0p

2. Install Dependencies

pip install -r requirements.txt

3. Run Flask API Locally

python app.py

Access the API at: http://localhost:5000

Supports /predict endpoint for fruit image classification.

4. CI/CD Pipeline

GitHub Actions automates training, Docker build, and deployment.

Logs for monitoring are visible locally.

5. MLflow Tracking

Start the MLflow UI (via Ngrok or locally) to track and visualize experiments, including runs, parameters, metrics, and artifacts.

6. Drift Monitoring

Run monitor_drift.py to continuously (CONTINUOUS_MONITORING = True) log metrics, top-class confidence, and class distributions.

Alerts for drift thresholds.

### Technologies Used

Python, TensorFlow/Keras

ResNet50V2 + Dense layers, BatchNorm, Dropout

MLflow (experiment tracking & model registry)

Git + DVC (versioning)

Docker + Flask (deployment)

GitHub Actions (CI/CD)
