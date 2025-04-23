#!/usr/bin/env python3
"""
Main script to run the Food Image Calorie Estimator pipeline.
This script provides a command-line interface for food classification and calorie estimation.
"""
import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from datasets import load_dataset
import matplotlib.pyplot as plt

# Import project modules
import config
from utils import set_tensorflow_cpu_only, load_image, normalize_food_name, calculate_calories, save_results
import calorie_estimator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Food Image Calorie Estimator')
    parser.add_argument('--image', type=str, help='Path to food image')
    parser.add_argument('--model', type=str, default='models/food_classifier_model.h5',
                        help='Path to trained model')
    parser.add_argument('--calories_csv', type=str, default='data/calories.csv',
                        help='Path to calories data CSV')
    parser.add_argument('--portion', type=float, default=100,
                        help='Portion size in grams (default: 100g)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to file')
    parser.add_argument('--train', action='store_true',
                        help='Train a new model instead of inference')
    return parser.parse_args()

def load_trained_model(model_path):
    """Load trained food classification model"""
    print(f"Loading model from {model_path}")
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        print(f"Model not found at {model_path}. Please train a model first.")
        return None

def get_class_names():
    """Get class names from Food-101 dataset"""
    print("Loading Food-101 class names...")
    dataset = load_dataset("ethz/food101", split="validation[:10]")  # Just load a small part for speed
    return dataset.features["label"].names

def predict_food(model, image_array, class_names):
    """Predict food class from image"""
    # Ensure image has batch dimension and right shape
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    predictions = model.predict(image_array)
    
    # Get top 5 predictions
    top_indices = np.argsort(predictions[0])[-5:][::-1]
    top_probs = predictions[0][top_indices]
    
    return top_indices, top_probs

def main():
    args = parse_arguments()
    
    # Force TensorFlow to use CPU only
    set_tensorflow_cpu_only()
    
    if args.train:
        # Import training module and train model
        print("Training mode selected. This will train a new model.")
        import food_classifier
        food_classifier.main()
        return
    
    # Load model
    model = load_trained_model(args.model)
    if model is None:
        return
    
    # Load class names
    class_names = get_class_names()
    
    # Load calorie estimator
    estimator = calorie_estimator.CalorieEstimator(args.calories_csv)
    
    # Load and process image
    if args.image and os.path.exists(args.image):
        image, image_array = load_image(args.image)
        
        # Predict food class
        indices, probs = predict_food(model, image_array, class_names)
        
        # Get top prediction
        top_class = class_names[indices[0]]
        top_prob = probs[0]
        
        # Normalize food name
        food_name = normalize_food_name(top_class)
        
        # Get calorie information
        calorie_info = estimator.direct_lookup(food_name)
        
        # Calculate total calories
        calories_per_100g = calorie_info.get('calories_per_100g', 0)
        if calories_per_100g is None:
            calories_per_100g = 0
        total_calories = calculate_calories(calories_per_100g, args.portion)
        
        # Display results
        print("\n===== Food Classification Results =====")
        print(f"Top prediction: {top_class.replace('_', ' ').title()} (Confidence: {top_prob:.2f})")
        
        print("\n===== Calorie Information =====")
        print(f"Matched food: {calorie_info.get('matched_food', 'Unknown')}")
        print(f"Calories per 100g: {calories_per_100g}")
        print(f"Portion size: {args.portion}g")
        print(f"Total calories: {total_calories:.1f}")
        
        # Save results if requested
        if args.save_results:
            prediction = {
                "food_name": top_class,
                "confidence": top_prob
            }
            
            calorie_result = {
                "calories_per_100g": calories_per_100g,
                "portion_size_grams": args.portion,
                "total_calories": total_calories
            }
            
            saved_path = save_results(image, prediction, calorie_result)
            print(f"\nResults saved to results/ directory")
        
        # Display image and predictions
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"{top_class.replace('_', ' ').title()}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(indices))
        plt.barh(y_pos, probs)
        plt.yticks(y_pos, [class_names[i].replace('_', ' ').title() for i in indices])
        plt.xlabel('Probability')
        plt.title('Top 5 Predictions')
        plt.tight_layout()
        plt.show()
    else:
        print("Please provide a valid image path using --image argument")
        print("Example: python main.py --image food_image.jpg --portion 150")

if __name__ == "__main__":
    main()