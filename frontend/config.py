#!/usr/bin/env python3
"""
Central configuration file for the Food Image Calorie Estimator project.
"""

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Dataset configuration
FOOD101_DATASET = "ethz/food101"
CALORIES_CSV = "calories.csv"

# Model configuration
MODEL_ARCHITECTURE = "EfficientNetB0"  # Options: "ResNet50", "EfficientNetB0", "MobileNetV2"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 101  # Food-101 has 101 classes

# Training configuration
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
VALIDATION_SPLIT = 0.2
USE_DATA_AUGMENTATION = True
USE_TRANSFER_LEARNING = True
FREEZE_BASE_MODEL = True

# DO NOT use CUDA as per project requirements
USE_CUDA = False

# Calorie estimation configuration
CALORIE_ESTIMATION_METHOD = "direct_lookup"  # Options: "direct_lookup", "regression"
DEFAULT_PORTION_SIZE = 100  # in grams

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1", "mae"]

# Streamlit app configuration
APP_TITLE = "Food Image Calorie Estimator"
APP_DESCRIPTION = "Upload a food image to identify the dish and estimate its caloric content"