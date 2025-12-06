Ripe & Unripe Fruit Classification MLOps Project

End-to-end MLOps project for classifying ripe vs unripe fruits using ResNet50V2 as the base model. Demonstrates a full ML lifecycle: experiment tracking, version control, automated CI/CD deployment, and production monitoring.

Key Features

Model Training & Experiment Tracking

Conducted experiments with Grid Search, Random Search, and Bayesian Optimization.

Tracked parameters, metrics, and artifacts using MLflow UI.

Model Registry & Deployment

Best model registered in MLflow Model Registry and promoted from Staging → Production.

Deployed via Dockerized Flask API with automated CI/CD using GitHub Actions.

Version Control & Reproducibility

Code, dataset, and models versioned using Git + DVC.

Monitoring & Drift Detection

Monitors model performance, top-class confidence, and class distribution.

Logs visible in MLflow UI; alerts trigger on drift or degradation.

Tech Stack

Python, TensorFlow/Keras, ResNet50V2, MLflow, DVC, Git, Docker, Flask, GitHub Actions

Quick Setup
git clone https://github.com/ssn-nishshanka/Fruit_Classification_MLOps.git
cd Fruit_Classification_MLOps
pip install -r requirements.txt
python app.py  # Run Flask API locally


Access the API at http://localhost:5000 using the /predict endpoint for fruit image classification.
