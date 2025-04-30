#!/usr/bin/env python3
"""
Utility functions for the Food Image Calorie Estimator project.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from datetime import datetime

def set_tensorflow_cpu_only():
    """Force TensorFlow to use CPU only (no CUDA)"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], 'GPU')
    physical_devices = tf.config.list_physical_devices('CPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    print("TensorFlow configured to use CPU only")

def load_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for model input"""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    return img, img_array

def normalize_food_name(food_name):
    """Normalize food name for better matching with calorie data"""
    if not isinstance(food_name, str):
        return ""
    # Convert to lowercase and replace underscores with spaces
    food_name = food_name.lower().replace('_', ' ')
    return food_name

def calculate_calories(calorie_per_100g, portion_size_grams):
    """Calculate total calories based on portion size"""
    return (calorie_per_100g * portion_size_grams) / 100

def create_visualization(image, predictions, class_names):
    """Create visualization for food classification results"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display original image
    ax1.imshow(image)
    ax1.set_title("Food Image")
    ax1.axis('off')
    
    # Display top predictions
    labels = [class_names[idx].replace('_', ' ').title() for idx in predictions[0][:5]]
    probs = predictions[1][:5]
    
    y_pos = np.arange(len(labels))
    ax2.barh(y_pos, probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title('Top 5 Predictions')
    
    plt.tight_layout()
    return fig

def save_results(image, prediction, calorie_info, timestamp=None):
    """Save prediction results to file"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save image
    image_path = f"results/food_image_{timestamp}.jpg"
    image.save(image_path)
    
    # Save prediction results
    results = {
        "timestamp": timestamp,
        "predicted_food": prediction["food_name"],
        "confidence": float(prediction["confidence"]),
        "calories_per_100g": float(calorie_info["calories_per_100g"]),
        "portion_size_grams": float(calorie_info["portion_size_grams"]),
        "total_calories": float(calorie_info["total_calories"]),
        "image_path": image_path
    }
    
    # Save as JSON
    with open(f"results/prediction_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return results