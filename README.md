# Ripe & Unripe Fruit Classification MLOps Project

End-to-end MLOps project for classifying ripe vs unripe fruits using **ResNet50V2** as the base model. Demonstrates a full ML lifecycle: experiment tracking, version control, automated CI/CD deployment, and production monitoring.

## Key Features

### Model Training & Experiment Tracking
- Conducted experiments with Manual Search, Grid Search, Random Search, and Bayesian Optimization.
- Tracked parameters, metrics, and artifacts using **MLflow UI**.

  <img width="1785" height="785" alt="image" src="https://github.com/user-attachments/assets/14b054fe-0557-4a03-8e54-6a65b0ccf723" />

### Model Registry & Deployment
- Best model registered in **MLflow Model Registry** and promoted from Staging → Production.
- Deployed via **Dockerized Flask API** with automated **CI/CD using GitHub Actions**.

### Version Control & Reproducibility
- Code, dataset, and models versioned using **Git + DVC**.

### Monitoring & Drift Detection
- **Model performance** – tracks metrics such as accuracy and loss over time.  
- **Top-class prediction confidence** – monitors how confident the model is in its most likely prediction for each input.  
- **Class distribution** – checks how often each class (ripe vs unripe) is predicted, helping detect data or prediction drift.
- Logs visible in **MLflow UI**; alerts trigger on drift or degradation.

## Tech Stack
**Python, TensorFlow/Keras, ResNet50V2, MLflow, DVC, Git, Docker, Flask, GitHub Actions**

## Quick Setup
```bash
git clone https://github.com/ssn-nishshanka/Fruit_Classification_MLOps.git
cd Fruit_Classification_MLOps
pip install -r requirements.txt
python app.py  # Run Flask API locally
```

## API Access
Once the Flask API is running, it can be accessed at [http://localhost:5000](http://localhost:5000) using the `/predict` endpoint for fruit image classification.
