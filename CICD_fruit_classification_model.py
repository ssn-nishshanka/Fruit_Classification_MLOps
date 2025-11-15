# In Colab, we ran multiple training loops (31 runs) to try different hyperparameters
# and log all experiments using MLflow. The best model was selected and saved.
#
# For CI/CD purposes, we only run one training session per pipeline run.

# The CI/CD training script trains a ResNet50V2-based fruit classifier once per pipeline run, logging all hyperparameters, data preprocessing, 
# augmentation, model architecture, and evaluation metrics to MLflow. The trained model is saved locally to MODEL_PATH for deployment.
# CI/CD automates the training workflow, ensuring the model runs correctly, logs metrics reliably, and is packaged for deployment automatically 
# whenever code changes, making the process reproducible and consistent.

#unzip dataset
import os
import zipfile
import shutil
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
ZIP_FILE = os.path.join(DATA_DIR, "fruit_dataset.zip")
EXTRACTED_DIR = os.path.join(DATA_DIR, "fruit_dataset")

os.makedirs(EXTRACTED_DIR, exist_ok=True)

SPLIT_DIR = os.path.join(DATA_DIR, "fruit_split_data")    # train/val/test folders
MODEL_PATH = "best_fruit_model_resnet50v2.h5"
MLFLOW_ARTIFACT_PATH = "mlruns"

os.makedirs(MLFLOW_ARTIFACT_PATH, exist_ok=True)

with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    for member in zip_ref.namelist():
        # remove top-level folder from path
        parts = member.split('/')
        filename = "/".join(parts[1:])  # skip the first folder
        if filename:  # skip empty strings
            dest_path = os.path.join(EXTRACTED_DIR, filename)
            if member.endswith('/'):
                os.makedirs(dest_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, 'wb') as f:
                    f.write(zip_ref.read(member))

print(f"Dataset extracted to {EXTRACTED_DIR}")


import tensorflow as tf
import mlflow

print("TensorFlow:", tf.__version__)
print("MLflow:", mlflow.__version__)

# Imports
import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, BackupAndRestore, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature

# mlflow setup
import mlflow
import mlflow.tensorflow
#from pyngrok import ngrok
#import subprocess

# Enable TensorFlow autologging
mlflow.tensorflow.autolog(log_models=True, log_datasets=False)

# Set MLflow tracking URI to a local folder (relative to repo)
import mlflow
import os

# Tracking directory
MLFLOW_ARTIFACT_PATH = "mlflow_runs"
os.makedirs(MLFLOW_ARTIFACT_PATH, exist_ok=True)

# Convert Windows path to valid file URI
tracking_uri = "file:///" + os.path.abspath(MLFLOW_ARTIFACT_PATH).replace("\\", "/")

mlflow.set_tracking_uri(tracking_uri)
print("Using tracking URI:", tracking_uri)

# Experiment setup
experiment_name = "Fruit_ResNet50V2"

if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(
        name=experiment_name,
        artifact_location=tracking_uri
    )

mlflow.set_experiment(experiment_name)
print(f"MLflow experiment '{experiment_name}' is set!")

# Retrieve experiment details
exp = mlflow.get_experiment_by_name(experiment_name)
print("Current Experiment:")
print(f"  ID: {exp.experiment_id}")
print(f"  Name: {exp.name}")
print(f"  Artifact Location: {exp.artifact_location}")

# Function to start an MLflow run with an auto-incremented run name based on previous runs
from mlflow.tracking import MlflowClient

def start_mlflow_run_auto(run_prefix="Run", nested=False):
    """
    Start a new MLflow run with an auto-incremented name.
    """
    experiment_name = "Fruit_ResNet50V2"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Use MlflowClient to fetch existing runs in the experiment
    client = MlflowClient()
    runs = client.search_runs([experiment.experiment_id])

    run_number = len(runs) + 1
    run_name = f"{run_prefix}_{run_number}"

    return mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id, nested=nested)

# paths for the dataset
source_dir = EXTRACTED_DIR
destination_dir = SPLIT_DIR 

# to list all class folder names inside the dataset
fruit_folders = sorted(os.listdir(source_dir))
print("Fruit Classes:", fruit_folders)

# Create directories and copy the relevant fruit folders
os.makedirs(destination_dir, exist_ok=True)

for folder in fruit_folders:
    src_folder = os.path.join(EXTRACTED_DIR, folder)
    dst_folder = os.path.join(destination_dir, folder)
    if os.path.isdir(src_folder):
        shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)


print("All fruit folders copied to:", destination_dir)

# Function to split a dataset into train/validation/test folders while preserving class structure
def split_dataset(source_dir, destination_dir, train_size=0.7, val_size=0.2, test_size=0.1):

    # check for floating-point sum
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
           "Train, val, and test sizes must sum to 1."

    splits = ['train', 'val', 'test']
    split_dirs = {split: os.path.join(destination_dir, split) for split in splits}

    # Create base split directories
    for dir_path in split_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Iterate over classes
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Create class folders inside each split folder
        for dir_path in split_dirs.values():
            os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

        # List images in class folder
        images = os.listdir(class_path)

        # Split into train and temp (val+test)
        train_imgs, temp_imgs = train_test_split(images, train_size=train_size, random_state=42)

        # Calculate relative val and test sizes
        val_ratio = val_size / (val_size + test_size)

        # Split temp into val and test
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=1 - val_ratio, random_state=42)

        # Copy images to their respective folders
        for img_list, split in zip([train_imgs, val_imgs, test_imgs], splits):
            for img in img_list:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_dirs[split], class_name, img)
                shutil.copy(src, dst)


# Splitting Dataset
#source_dir = '/content/fruit_data'           # folder with copied data
#destination_dir = '/content/fruit_split_data'  # folder where the train/val/test splits will be created

# paths for the dataset
source_dir = EXTRACTED_DIR
destination_dir = SPLIT_DIR 

split_dataset(source_dir, destination_dir)

TRAIN_DIR = os.path.join(destination_dir, 'train')
VAL_DIR = os.path.join(destination_dir, 'val')
TEST_DIR = os.path.join(destination_dir, 'test')

# Class Distribution in the train split
#split_dataset_path = '/content/fruit_split_data/train'  # the train folder path
split_dataset_path = TRAIN_DIR
class_names = [folder for folder in os.listdir(split_dataset_path)
               if os.path.isdir(os.path.join(split_dataset_path, folder))]

for class_name in class_names:
    class_path = os.path.join(split_dataset_path, class_name)
    num_images = len(os.listdir(class_path))
    print(f"{class_name}: {num_images} images")


# Data Generators Setup for Raw vs Ripe Fruit
target_size = (224, 224)
batch_size = 30
num_classes = 22   # 11 raw + 11 ripe = 22 classes
channels = 3
epochs = 100

# data augmentation - this helps us to increase the size of the dataset and introduce variability in the dataset.
# data augmentation with keras by using the ImageDataGenerator class.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=(0.4, 1.6),
    fill_mode='nearest',
    channel_shift_range=40.0,
    shear_range=25.0
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.7, 1.3]
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.7, 1.3]
)

train_generator = train_datagen.flow_from_directory(
    #'/content/fruit_split_data/train',
    TRAIN_DIR,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    #'/content/fruit_split_data/val',
    VAL_DIR,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    #'/content/fruit_split_data/test',
    TEST_DIR,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Check the train/val/test directories and print how many images each class contains in every split

import os

base_path = SPLIT_DIR

# Check each split directory
for split in ['train', 'val', 'test']:
    split_path = os.path.join(base_path, split)
    print(f"\nContents of: {split_path}")

    if not os.path.exists(split_path):
        print(f" {split_path} does NOT exist!")
        continue

    class_folders = [folder for folder in os.listdir(split_path)
                     if os.path.isdir(os.path.join(split_path, folder))]

    print(f"Found {len(class_folders)} class folders.")

    for class_name in sorted(class_folders):
        class_path = os.path.join(split_path, class_name)
        num_images = len(os.listdir(class_path))
        print(f"   - {class_name}: {num_images} images")

# Calculate balanced class weights from the training data to handle class imbalance during model training

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights using train_generator
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

# Convert to dictionary format
class_weight_dict = dict(enumerate(class_weights))

# Print the class weights
print("Class Weights per Class Index:")
for class_index, weight in class_weight_dict.items():
    class_name = list(train_generator.class_indices.keys())[list(train_generator.class_indices.values()).index(class_index)]
    print(f"  {class_index} ({class_name}): {weight:.4f}")


# Gradually unfreezes deeper layers of the base model during training
class GradualUnfreezing(tf.keras.callbacks.Callback):
    def __init__(self, base_model, layers_per_unfreeze=5, start_epoch=3, interval=3):
        super().__init__()
        self.base_model = base_model
        self.layers_per_unfreeze = layers_per_unfreeze
        self.start_epoch = start_epoch
        self.interval = interval
        self.unfrozen_layers = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch and (epoch - self.start_epoch) % self.interval == 0:
            total_layers = len(self.base_model.layers)
            next_unfreeze = self.unfrozen_layers + self.layers_per_unfreeze
            if next_unfreeze <= total_layers:
                for layer in self.base_model.layers[-next_unfreeze: -self.unfrozen_layers or None]:
                    layer.trainable = True
                self.unfrozen_layers = next_unfreeze
                print(f"\n[Gradual Unfreezing] Unfrozen layers: last {self.unfrozen_layers} of {total_layers}")

# model architecture
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers, regularizers, Model, Input

def build_model(num_classes):
    # Initialize ResNet50V2 base
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Input layer
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)

    # Global pooling and dropout
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Residual block 1
    x1 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.3)(x1)
    x_res = layers.Dense(1024, activation='linear')(x)
    x = layers.Add()([x_res, x1])
    x = layers.Activation('relu')(x)

    # Residual block 2
    x2 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x2 = layers.Dropout(0.3)(x2)
    x_res2 = layers.Dense(512, activation='linear')(x)
    x = layers.Add()([x_res2, x2])
    x = layers.Activation('relu')(x)

    # Residual block 3
    x3 = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x3 = layers.Dropout(0.3)(x3)
    x_res3 = layers.Dense(256, activation='linear')(x)
    x = layers.Add()([x_res3, x3])
    x = layers.Activation('relu')(x)

    # Final output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Return fresh model
    return Model(inputs, outputs)

# Load ResNet50V2 as the feature extractor and build the final classification model
base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(target_size[0], target_size[1], channels)
)

model = build_model(num_classes)

# Learning Rate Schedule

initial_learning_rate = 1e-4
warmup_epochs = 5
total_epochs = epochs

def warmup_cosine_decay_schedule(epoch):
    if epoch < warmup_epochs:
        return initial_learning_rate * ((epoch + 1) / warmup_epochs)
    else:
        return initial_learning_rate * (
            0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        )
    
# Save checkpoints and backup to Google Drive
#checkpoint_dir = "/content/drive/MyDrive/fruit_checkpoints"
#backup_dir = "/content/drive/MyDrive/fruit_backup"
#tensorboard_log_dir = "/content/drive/MyDrive/fruit_tensorboard_logs"

#os.makedirs(checkpoint_dir, exist_ok=True)
#os.makedirs(backup_dir, exist_ok=True)
#os.makedirs(tensorboard_log_dir, exist_ok=True)

import os

# Base model folder
MODEL_DIR = "models"

# Subfolders
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
BACKUP_DIR = os.path.join(MODEL_DIR, "backup")
TENSORBOARD_LOG_DIR = os.path.join(MODEL_DIR, "tensorboard_logs")

# Create folders if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


# Callbacks

callbacks = [
    # Stop training early if no improvement
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        min_delta=1e-4,
        verbose=1
    ),

    # Save the best model based on validation accuracy
    tf.keras.callbacks.ModelCheckpoint(
        #filepath=os.path.join(checkpoint_dir, 'best_fruit_model_resnet50v2.keras'),
        filepath=os.path.join(CHECKPOINT_DIR, 'best_fruit_model_resnet50v2.keras'),  # relative path
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    ),

    # Apply custom warmup and cosine decay learning rate
    tf.keras.callbacks.LearningRateScheduler(warmup_cosine_decay_schedule),

    # Gradually unfreeze base model layers
    GradualUnfreezing(base_model),

    # Automatically back up training in case of interruption
    #tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),

    # Log training for TensorBoard visualization
    #tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
]

## 1) Manual Search - optimizer tuning

# Compile Model

# 1. AdamW
#from tensorflow.keras.optimizers import AdamW
#from tensorflow.keras.metrics import AUC, Precision, Recall

# Optimizer with weight decay
#optimizer = AdamW(
#    learning_rate=initial_learning_rate,
#    weight_decay=1e-4
#)

# 2. SGD (Momentum-based)
#from tensorflow.keras.optimizers import SGD

#optimizer = SGD(
#    learning_rate=initial_learning_rate,
#    momentum=0.9,
#    nesterov=True
#)

# 3. RMSprop (Hybrid adaptive)
from tensorflow.keras.optimizers import RMSprop

optimizer = RMSprop(
   learning_rate=initial_learning_rate,
    rho=0.9,
    momentum=0.9
)

# according to the plots logged and visualized in mlflow ui the RMSprop optimizer is comparatively is better because the training and validation curves are closely aligned. This indicates stable learning and
# good generalization. no prominent overfitting or underfitting as plots obtained from training with the other optimizers.

# 4. Nadam (Hybrid Adaptive)
#from tensorflow.keras.optimizers import Nadam
#from tensorflow.keras.metrics import AUC, Precision, Recall

#optimizer = Nadam(
#    learning_rate=initial_learning_rate,
#   beta_1=0.9,
#    beta_2=0.999,
#    epsilon=1e-7
#)

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# summary of model architecture
model.summary()

# Show one sample image per class by looping through batches until all class labels are displayed once in a grid layout

import math

# Get class names from train_generator
class_names = [k for k, v in sorted(train_generator.class_indices.items(), key=lambda item: item[1])]

# Set up the grid with 3 rows
n_rows = 3
n_cols = math.ceil(len(class_names) / n_rows)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = axes.flatten()  # Flatten to 1D list for easy indexing

# Track displayed classes
displayed_classes = set()

# Loop until we have one image per class
while len(displayed_classes) < len(class_names):
    images, labels = next(train_generator)
    for i in range(len(images)):
        label_idx = np.argmax(labels[i])
        if label_idx not in displayed_classes:
            axes[label_idx].imshow(images[i])
            axes[label_idx].set_title(f"{class_names[label_idx]}")
            axes[label_idx].axis('off')
            displayed_classes.add(label_idx)
        if len(displayed_classes) == len(class_names):
            break

# Hide any extra subplots (if total grid > number of classes)
for j in range(len(class_names), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# Display 12 augmented images from the training set in a 4x4 grid with their class labels
plt.figure(figsize=(20, 15))

for i in range(12):
    if i >= len(images):
        break
    ax = plt.subplot(3, 4, i + 1)  # 4 rows, 4 columns = 12 images
    img = images[i] * 255.0
    plt.imshow(img.astype("uint8"))
    plt.title(class_names[np.argmax(labels[i])])
    plt.axis('off')

plt.show()

# function to log training curves, confusion matrix, and classification report to MLflow using the trained model, its history, and the data generator.

def log_custom_metrics(history, model, generator):
    # Accuracy/Loss plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label='train_acc')
    ax[0].plot(history.history['val_accuracy'], label='val_acc')
    ax[0].legend(); ax[0].set_title("Accuracy")

    ax[1].plot(history.history['loss'], label='train_loss')
    ax[1].plot(history.history['val_loss'], label='val_loss')
    ax[1].legend(); ax[1].set_title("Loss")

    plt.savefig("training_curves.png")
    mlflow.log_artifact("training_curves.png", artifact_path="plots")
    plt.close(fig)

    # Predictions for Confusion Matrix
    y_true = generator.classes
    y_pred = np.argmax(model.predict(generator), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=generator.class_indices.keys())
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=generator.class_indices.keys())
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt", artifact_path="reports")

# MLflow run block that trains the model and logs all training details, metrics, and the saved model.
with start_mlflow_run_auto() as run:
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # Log preprocessing and augmentation info
    mlflow.log_param("normalization", "rescale=1./255")
    mlflow.log_param(
        "train_augmentation",
        "rotation_range=50, width_shift_range=0.3, height_shift_range=0.3, zoom_range=0.3, horizontal_flip=True, brightness_range=(0.4,1.6), fill_mode='nearest', channel_shift_range=40.0, shear_range=25.0"
    )
    mlflow.log_param("target_size", "(224, 224)")
    mlflow.log_param("num_classes", 22)

    # Log model architecture info
    mlflow.log_param(
        "model_architecture",
        "ResNet50V2 base + 3 residual dense blocks + dropout + batchnorm"
    )
    mlflow.log_param("base_model", "ResNet50V2 (imagenet, include_top=False)")
    mlflow.log_param("frozen_layers", "All layers frozen initially")
    mlflow.log_param("activation_functions", "ReLU for hidden layers, Softmax for output")
    mlflow.log_param("regularization", "L2(1e-5)")

    # Log hyperparameters

    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

    # Log optimizer details from model
    opt_config = model.optimizer.get_config()
    mlflow.log_param("optimizer", model.optimizer.__class__.__name__)
    for k, v in opt_config.items():
        if isinstance(v, (dict, list)):
            v = str(v)
        mlflow.log_param(f"optimizer_{k}", v)

    # Log learning rate schedule parameters
    mlflow.log_param("initial_learning_rate", initial_learning_rate)
    mlflow.log_param("warmup_epochs", warmup_epochs)
    mlflow.log_param("total_epochs", total_epochs)
    mlflow.log_param("lr_schedule", "warmup_cosine_decay_schedule")

    # Evaluate test set
    test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(test_generator, verbose=1)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)

    # Log artifacts (plots + metrics)
    log_custom_metrics(history, model, test_generator)

    # Log model & register in MLflow Registry
    try:
        mlflow.keras.log_model(
            model,
            artifact_path="fruit_classifier_model",
            registered_model_name="FruitClassifier"
        )
        print("Model registered successfully in MLflow Model Registry!")
    except MlflowException as e:
        print("Model registration failed:", e)

    print("Run completed with ID:", run.info.run_id)

# save model

import os

# Path to save model
#save_path = "/content/best_fruit_model_resnet50v2.h5"

# Save entire model
#model.save(save_path)
#print(f"Model saved at {save_path}")

model.save(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")