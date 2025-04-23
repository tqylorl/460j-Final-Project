#!/usr/bin/env python3
# model_evaluation.py
"""
Comprehensive evaluation script for food classification and calorie estimation system.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import logging
from pathlib import Path
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_food101_test_dataset():
    """Load the Food-101 test dataset."""
    logger.info("Loading Food-101 test dataset...")
    ds = load_dataset("ethz/food101")
    test_dataset = ds["test"]
    logger.info(f"Loaded {len(test_dataset)} test samples")
    return test_dataset

def load_trained_model(model_path):
    """
    Load the trained food classification model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        The loaded model and device
    """
    logger.info(f"Loading trained model from {model_path}...")
    
    # Assume we're using a pretrained ResNet50 with the final layer modified for our task
    model = models.resnet50(pretrained=False)
    num_classes = 101  # Food-101 has 101 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load the trained weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    return model, device

def load_calorie_mapping(csv_path):
    """
    Load the calorie mapping from CSV file.
    
    Args:
        csv_path: Path to the CSV file with calorie information
        
    Returns:
        DataFrame with the calorie information
    """
    logger.info(f"Loading calorie mapping from {csv_path}...")
    try:
        calorie_df = pd.read_csv(csv_path)
        logger.info(f"Loaded calorie information for {len(calorie_df)} food items")
        return calorie_df
    except Exception as e:
        logger.error(f"Error loading calorie mapping: {e}")
        raise

def map_food101_to_calorie_data(food101_classes, calorie_df):
    """
    Create a mapping between Food-101 classes and calorie data.
    
    Args:
        food101_classes: List of Food-101 class names
        calorie_df: DataFrame with calorie information
        
    Returns:
        Dictionary mapping Food-101 class names to calorie values
    """
    logger.info("Mapping Food-101 classes to calorie data...")
    calorie_mapping = {}
    
    # Normalize food names for better matching
    food101_classes_norm = [c.replace('_', ' ').lower() for c in food101_classes]
    calorie_df['FoodItem_norm'] = calorie_df['FoodItem'].str.lower()
    
    # Count matches
    matches = 0
    
    for i, food_class in enumerate(food101_classes):
        food_class_norm = food101_classes_norm[i]
        
        # Try exact match
        exact_match = calorie_df[calorie_df['FoodItem_norm'] == food_class_norm]
        
        if len(exact_match) > 0:
            # Use the first match if multiple are found
            calorie_value = float(exact_match.iloc[0]['Cals_per100grams'])
            calorie_mapping[food_class] = calorie_value
            matches += 1
        else:
            # Try fuzzy match (contains)
            fuzzy_matches = calorie_df[calorie_df['FoodItem_norm'].str.contains(food_class_norm, na=False)]
            
            if len(fuzzy_matches) > 0:
                # Use the average if multiple are found
                calorie_values = fuzzy_matches['Cals_per100grams'].astype(float)
                avg_calorie = calorie_values.mean()
                calorie_mapping[food_class] = avg_calorie
                matches += 1
            else:
                # Try match by category
                category_matches = calorie_df[calorie_df['FoodCategory'].str.lower().str.contains(food_class_norm, na=False)]
                
                if len(category_matches) > 0:
                    # Use the average calories for the category
                    calorie_values = category_matches['Cals_per100grams'].astype(float)
                    avg_calorie = calorie_values.mean()
                    calorie_mapping[food_class] = avg_calorie
                    matches += 1
                else:
                    logger.warning(f"No calorie data found for {food_class}")
                    # Fallback to a median value from the dataset
                    calorie_mapping[food_class] = calorie_df['Cals_per100grams'].astype(float).median()
    
    logger.info(f"Mapped {matches} out of {len(food101_classes)} food classes to calorie data")
    return calorie_mapping

def get_image_transform():
    """
    Get the image transformation pipeline for model evaluation.
    
    Returns:
        torchvision transforms composer
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def evaluate_classification(model, device, test_dataset, batch_size=32):
    """
    Evaluate the classification performance of the model.
    
    Args:
        model: The trained model
        device: The device to run inference on
        test_dataset: The test dataset
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with classification evaluation metrics and predictions
    """
    logger.info("Evaluating classification performance...")
    
    # Create data loader for test data
    transform = get_image_transform()
    
    # Store predictions and ground truth
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Process in batches
    num_samples = len(test_dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_samples = test_dataset[start_idx:end_idx]
            
            # Convert images to tensors
            batch_images = []
            batch_labels = []
            
            for sample in batch_samples:
                img = sample['image']
                label = sample['label']
                
                # Convert PIL image to tensor
                img_tensor = transform(img).unsqueeze(0)
                batch_images.append(img_tensor)
                batch_labels.append(label)
            
            # Stack batch tensors
            batch_images = torch.cat(batch_images).to(device)
            batch_labels = np.array(batch_labels)
            
            # Get model predictions
            outputs = model(batch_images)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predicted_classes = np.argmax(probabilities, axis=1)
            
            all_preds.extend(predicted_classes)
            all_labels.extend(batch_labels)
            all_probs.append(probabilities)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{num_batches} batches")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    logger.info(f"Classification Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Precision: {precision:.4f}")
    logger.info(f"Classification Recall: {recall:.4f}")
    logger.info(f"Classification F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def evaluate_calorie_estimation(class_predictions, true_labels, calorie_mapping, class_names):
    """
    Evaluate the calorie estimation performance.
    
    Args:
        class_predictions: Predicted class indices
        true_labels: True class indices
        calorie_mapping: Dictionary mapping class names to calorie values
        class_names: List of class names
        
    Returns:
        Dictionary with calorie estimation evaluation metrics
    """
    logger.info("Evaluating calorie estimation performance...")
    
    # Convert class indices to class names
    pred_classes = [class_names[idx] for idx in class_predictions]
    true_classes = [class_names[idx] for idx in true_labels]
    
    # Get predicted and actual calorie values
    pred_calories = np.array([calorie_mapping.get(cls, 0) for cls in pred_classes])
    true_calories = np.array([calorie_mapping.get(cls, 0) for cls in true_classes])
    
    # Calculate metrics
    mae = mean_absolute_error(true_calories, pred_calories)
    rmse = np.sqrt(mean_squared_error(true_calories, pred_calories))
    
    logger.info(f"Calorie Estimation MAE: {mae:.2f} calories per 100g")
    logger.info(f"Calorie Estimation RMSE: {rmse:.2f} calories per 100g")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'predicted_calories': pred_calories,
        'true_calories': true_calories,
        'predicted_classes': pred_classes,
        'true_classes': true_classes
    }

def visualize_classification_results(eval_results, class_names, output_dir='./results'):
    """
    Visualize the classification results.
    
    Args:
        eval_results: Dictionary with classification evaluation results
        class_names: List of class names
        output_dir: Directory to save visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info("Visualizing classification results...")
    
    # Create a confusion matrix heatmap for most problematic classes
    plt.figure(figsize=(16, 14))
    conf_matrix = eval_results['confusion_matrix']
    
    # Calculate class-wise accuracy and find most problematic classes
    class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    worst_20_indices = np.argsort(class_accuracy)[:20]
    
    # Create subset confusion matrix for worst classes
    worst_conf_matrix = conf_matrix[worst_20_indices][:, worst_20_indices]
    worst_class_names = [class_names[i] for i in worst_20_indices]
    
    # Plot confusion matrix for worst classes
    sns.heatmap(worst_conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=worst_class_names, yticklabels=worst_class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for 20 Most Challenging Classes')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'worst_classes_confusion.png'), dpi=300)
    plt.close()
    
    # Plot class-wise accuracy bar chart
    sorted_indices = np.argsort(class_accuracy)
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(sorted_indices)), class_accuracy[sorted_indices])
    plt.yticks(range(len(sorted_indices)), [class_names[i] for i in sorted_indices])
    plt.xlabel('Accuracy')
    plt.title('Classification Accuracy by Food Class')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_accuracy.png'), dpi=300)
    plt.close()
    
    # Plot worst 10 classes
    worst_10_indices = sorted_indices[:10]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(worst_10_indices)), class_accuracy[worst_10_indices])
    plt.yticks(range(len(worst_10_indices)), [class_names[i] for i in worst_10_indices])
    plt.xlabel('Accuracy')
    plt.title('10 Worst Performing Food Classes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'worst_10_classes.png'), dpi=300)
    plt.close()
    
    # Plot best 10 classes
    best_10_indices = sorted_indices[-10:][::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(best_10_indices)), class_accuracy[best_10_indices])
    plt.yticks(range(len(best_10_indices)), [class_names[i] for i in best_10_indices])
    plt.xlabel('Accuracy')
    plt.title('10 Best Performing Food Classes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_10_classes.png'), dpi=300)
    plt.close()
    
    # Plot overall metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [eval_results[m] for m in metrics]
    
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.title('Overall Classification Performance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics.png'), dpi=300)
    plt.close()
    
    logger.info(f"Classification visualizations saved to {output_dir}")

def visualize_calorie_estimation(calorie_results, output_dir='./results'):
    """
    Visualize the calorie estimation results.
    
    Args:
        calorie_results: Dictionary with calorie estimation results
        output_dir: Directory to save visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info("Visualizing calorie estimation results...")
    
    # Scatter plot of predicted vs actual calories
    true_calories = calorie_results['true_calories']
    pred_calories = calorie_results['predicted_calories']
    
    plt.figure(figsize=(10, 8))
    plt.scatter(true_calories, pred_calories, alpha=0.5)
    
    # Add perfect prediction line
    max_cal = max(np.max(true_calories), np.max(pred_calories))
    min_cal = min(np.min(true_calories), np.min(pred_calories))
    plt.plot([min_cal, max_cal], [min_cal, max_cal], 'r--')
    
    plt.xlabel('Actual Calories (per 100g)')
    plt.ylabel('Predicted Calories (per 100g)')
    plt.title('Predicted vs Actual Calorie Values')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'calorie_prediction.png'), dpi=300)
    plt.close()
    
    # Histogram of calorie estimation errors
    errors = pred_calories - true_calories
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error (Calories per 100g)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Calorie Prediction Errors')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'calorie_error_distribution.png'), dpi=300)
    plt.close()
    
    # Error vs true calorie value
    plt.figure(figsize=(10, 8))
    plt.scatter(true_calories, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Calories (per 100g)')
    plt.ylabel('Prediction Error (Calories per 100g)')
    plt.title('Calorie Prediction Error vs Actual Calorie Value')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'calorie_error_vs_value.png'), dpi=300)
    plt.close()
    
    # Boxplot of errors by high/medium/low calorie ranges
    # Define calorie ranges
    calorie_ranges = [
        (0, 100, 'Low (0-100)'),
        (100, 250, 'Medium (100-250)'),
        (250, float('inf'), 'High (>250)')
    ]
    
    range_errors = []
    range_labels = []
    
    for low, high, label in calorie_ranges:
        mask = (true_calories >= low) & (true_calories < high)
        if np.sum(mask) > 0:
            range_errors.append(errors[mask])
            range_labels.append(label)
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(range_errors, labels=range_labels)
    plt.ylabel('Prediction Error (Calories per 100g)')
    plt.title('Calorie Prediction Error by Calorie Range')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'calorie_error_by_range.png'), dpi=300)
    plt.close()
    
    logger.info(f"Calorie estimation visualizations saved to {output_dir}")

def analyze_challenging_cases(class_results, calorie_results, class_names, test_dataset, calorie_mapping, output_dir='./results'):
    """
    Analyze the most challenging cases for the model.
    
    Args:
        class_results: Dictionary with classification results
        calorie_results: Dictionary with calorie estimation results
        class_names: List of class names
        test_dataset: The test dataset
        calorie_mapping: Dictionary mapping class names to calorie values
        output_dir: Directory to save analysis
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info("Analyzing challenging cases...")
    
    # Get misclassified samples
    predictions = class_results['predictions']
    true_labels = class_results['labels']
    probs = class_results['probabilities']
    
    misclassified_indices = np.where(predictions != true_labels)[0]
    
    if len(misclassified_indices) == 0:
        logger.info("No misclassified samples found!")
        return
    
    # Get the confidence scores for the predicted class
    misclassified_confidences = np.max(probs[misclassified_indices], axis=1)
    
    # Sort by confidence (higher confidence on wrong prediction is more interesting)
    sorted_indices = misclassified_indices[np.argsort(-misclassified_confidences)]
    
    # Analyze top 10 most confident misclassifications
    analysis_file = os.path.join(output_dir, 'challenging_cases_analysis.txt')
    
    with open(analysis_file, 'w') as f:
        f.write("ANALYSIS OF CHALLENGING CASES\n")
        f.write("============================\n\n")
        
        f.write("TOP 10 MOST CONFIDENT MISCLASSIFICATIONS\n")
        f.write("--------------------------------------\n\n")
        
        for i, idx in enumerate(sorted_indices[:10]):
            true_class = class_names[true_labels[idx]]
            pred_class = class_names[predictions[idx]]
            confidence = np.max(probs[idx])
            
            true_calorie = calorie_mapping.get(true_class, 0)
            pred_calorie = calorie_mapping.get(pred_class, 0)
            calorie_error = pred_calorie - true_calorie
            
            f.write(f"Example {i+1}:\n")
            f.write(f"  True Class: {true_class}\n")
            f.write(f"  Predicted Class: {pred_class}\n")
            f.write(f"  Confidence: {confidence:.4f}\n")
            f.write(f"  True Calories: {true_calorie:.1f} per 100g\n")
            f.write(f"  Predicted Calories: {pred_calorie:.1f} per 100g\n")
            f.write(f"  Calorie Error: {calorie_error:.1f} per 100g\n\n")
        
        # Analyze the most challenging classes (lowest accuracy)
        confusion = class_results['confusion_matrix']
        class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)
        
        worst_classes_idx = np.argsort(class_accuracy)[:10]
        
        f.write("\nMOST CHALLENGING FOOD CLASSES\n")
        f.write("---------------------------\n\n")
        
        for i, class_idx in enumerate(worst_classes_idx):
            class_name = class_names[class_idx]
            acc = class_accuracy[class_idx]
            
            # Find top confusions for this class
            confusion_row = confusion[class_idx].copy()
            confusion_row[class_idx] = 0  # Zero out the diagonal
            top_confusions_idx = np.argsort(-confusion_row)[:3]
            
            f.write(f"{i+1}. {class_name}\n")
            f.write(f"   Accuracy: {acc:.4f}\n")
            f.write("   Top confusions:\n")
            
            for conf_idx in top_confusions_idx:
                if confusion_row[conf_idx] > 0:
                    conf_class = class_names[conf_idx]
                    conf_count = confusion_row[conf_idx]
                    conf_percent = conf_count / np.sum(confusion[class_idx]) * 100
                    f.write(f"     - {conf_class}: {conf_count} instances ({conf_percent:.1f}%)\n")
            
            # Calorie implications
            true_cal = calorie_mapping.get(class_name, 0)
            top_conf_cal = calorie_mapping.get(class_names[top_confusions_idx[0]], 0)
            cal_diff = top_conf_cal - true_cal
            f.write(f"   Calorie impact: Most common confusion leads to {cal_diff:.1f} calories difference\n\n")
        
        # Analyze the most difficult calorie estimations
        calorie_errors = np.abs(calorie_results['predicted_calories'] - calorie_results['true_calories'])
        worst_calorie_idx = np.argsort(-calorie_errors)[:10]
        
        f.write("\nLARGEST CALORIE ESTIMATION ERRORS\n")
        f.write("-------------------------------\n\n")
        
        for i, idx in enumerate(worst_calorie_idx):
            true_class = calorie_results['true_classes'][idx]
            pred_class = calorie_results['predicted_classes'][idx]
            true_cal = calorie_results['true_calories'][idx]
            pred_cal = calorie_results['predicted_calories'][idx]
            error = calorie_errors[idx]
            
            f.write(f"{i+1}. Sample with {true_class}\n")
            f.write(f"   Predicted as: {pred_class}\n")
            f.write(f"   True Calories: {true_cal:.1f} per 100g\n")
            f.write(f"   Predicted Calories: {pred_cal:.1f} per 100g\n")
            f.write(f"   Absolute Error: {error:.1f} per 100g\n\n")
    
    # Generate visualization of high-confidence misclassifications by error magnitude
    plt.figure(figsize=(12, 8))
    
    # Filter to just misclassified samples
    misclass_true_classes = calorie_results['true_classes'][misclassified_indices]
    misclass_pred_classes = calorie_results['predicted_classes'][misclassified_indices]
    misclass_confidences = np.max(probs[misclassified_indices], axis=1)
    
    # Calculate calorie errors for misclassified samples
    misclass_true_cals = np.array([calorie_mapping.get(c, 0) for c in misclass_true_classes])
    misclass_pred_cals = np.array([calorie_mapping.get(c, 0) for c in misclass_pred_classes])
    misclass_cal_errors = np.abs(misclass_pred_cals - misclass_true_cals)
    
    # Plot error vs confidence
    plt.scatter(misclass_confidences, misclass_cal_errors, alpha=0.5)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Absolute Calorie Error (per 100g)')
    plt.title('Confidence vs Calorie Error for Misclassified Samples')
    plt.grid(True)
    
    # Add annotations for worst points
    worst_points = np.argsort(-misclass_cal_errors)[:5]
    for i in worst_points:
        plt.annotate(
            f"{misclass_true_classes[i]} → {misclass_pred_classes[i]}",
            (misclass_confidences[i], misclass_cal_errors[i]),
            xytext=(10, 10),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2')
        )
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_confidence.png'), dpi=300)
    plt.close()
    
    logger.info(f"Challenging cases analysis saved to {analysis_file}")

def generate_summary_report(class_results, calorie_results, output_dir='./results'):
    """
    Generate a summary report of the evaluation.
    
    Args:
        class_results: Dictionary with classification results
        calorie_results: Dictionary with calorie estimation results
        output_dir: Directory to save the report
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    report_file = os.path.join(output_dir, 'evaluation_summary.txt')
    
    with open(report_file, 'w') as f:
        f.write("FOOD CLASSIFICATION AND CALORIE ESTIMATION EVALUATION SUMMARY\n")
        f.write("========================================================\n\n")
        
        f.write("CLASSIFICATION PERFORMANCE\n")
        f.write("------------------------\n")
        f.write(f"Accuracy: {class_results['accuracy']:.4f}\n")
        f.write(f"Precision: {class_results['precision']:.4f}\n")
        f.write(f"Recall: {class_results['recall']:.4f}\n")
        f.write(f"F1 Score: {class_results['f1']:.4f}\n\n")
        
        f.write("CALORIE ESTIMATION PERFORMANCE\n")
        f.write("----------------------------\n")
        f.write(f"Mean Absolute Error: {calorie_results['mae']:.2f} calories per 100g\n")
        f.write(f"Root Mean Squared Error: {calorie_results['rmse']:.2f} calories per 100g\n\n")
        
        # Additional statistics
        errors = calorie_results['predicted_calories'] - calorie_results['true_calories']
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        
        f.write("ADDITIONAL STATISTICS\n")
        f.write("-------------------\n")
        f.write(f"Mean Calorie Estimation Error: {mean_error:.2f} calories per 100g\n")
        f.write(f"Median Calorie Estimation Error: {median_error:.2f} calories per 100g\n")
        
        # Over/Under estimation analysis
        over_est = np.sum(errors > 0)
        under_est = np.sum(errors < 0)
        over_percent = over_est / len(errors) * 100
        under_percent = under_est / len(errors) * 100
        
        f.write(f"Over-estimations: {over_est} samples ({over_percent:.1f}%)\n")
        f.write(f"Under-estimations: {under_est} samples ({under_percent:.1f}%)\n\n")
        
        # Error analysis by actual calorie value
        calorie_bins = [(0, 100), (100, 200), (200, 300), (300, 400), (400, float('inf'))]
        f.write("ERROR ANALYSIS BY CALORIE RANGE\n")
        f.write("-----------------------------\n")
        
        for low, high in calorie_bins:
            range_label = f"{low}-{high if high != float('inf') else '+'}"
            mask = (calorie_results['true_calories'] >= low) & (calorie_results['true_calories'] < high)
            
            if np.sum(mask) > 0:
                range_errors = errors[mask]
                range_mae = np.mean(np.abs(range_errors))
                range_rmse = np.sqrt(np.mean(np.square(range_errors)))
                
                f.write(f"Calorie range {range_label}:\n")
                f.write(f"  Samples: {np.sum(mask)}\n")
                f.write(f"  MAE: {range_mae:.2f} calories per 100g\n")
                f.write(f"  RMSE: {range_rmse:.2f} calories per 100g\n\n")
        
        f.write("VISUALIZATIONS\n")
        f.write("-------------\n")
        f.write("The following visualizations have been generated:\n")
        f.write("- worst_classes_confusion.png: Confusion matrix for most challenging classes\n")
        f.write("- class_accuracy.png: Accuracy by food class\n")
        f.write("- worst_10_classes.png: 10 worst performing food classes\n")
        f.write("- best_10_classes.png: 10 best performing food classes\n")
        f.write("- overall_metrics.png: Overall classification metrics\n")
        f.write("- calorie_prediction.png: Predicted vs actual calories\n")
        f.write("- calorie_error_distribution.png: Distribution of calorie prediction errors\n")
        f.write("- calorie_error_vs_value.png: Calorie prediction error vs actual calorie value\n")
        f.write("- calorie_error_by_range.png: Box plot of errors by calorie range\n")
        f.write("- error_vs_confidence.png: Confidence vs calorie error for misclassifications\n\n")
        
        f.write("DETAILED ANALYSIS\n")
        f.write("----------------\n")
        f.write("See 'challenging_cases_analysis.txt' for a detailed analysis of challenging cases,\n")
        f.write("including the most confident misclassifications, most challenging food classes,\n")
        f.write("and largest calorie estimation errors.\n")
    
    # Also save results in JSON format for easy programmatic access
    json_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'classification': {
            'accuracy': float(class_results['accuracy']),
            'precision': float(class_results['precision']),
            'recall': float(class_results['recall']),
            'f1': float(class_results['f1'])
        },
        'calorie_estimation': {
            'mae': float(calorie_results['mae']),
            'rmse': float(calorie_results['rmse']),
            'mean_error': float(mean_error),
            'median_error': float(median_error)
        }
    }
    
    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Summary report saved to {report_file}")

def main():
    # Configuration
    model_path = './models/food_classification_model.pth'  # Path to saved model
    csv_path = 'calories.csv'  # Path to calorie data
    output_dir = './evaluation_results'  # Directory to save results
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(output_dir, 'evaluation.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting model evaluation")
    
    try:
        # Load test dataset
        test_dataset = load_food101_test_dataset()
        class_names = test_dataset.features['label'].names
        
        # Load trained model
        model, device = load_trained_model(model_path)
        
        # Load calorie mapping
        calorie_df = load_calorie_mapping(csv_path)
        
        # Map Food-101 classes to calorie data
        calorie_mapping = map_food101_to_calorie_data(class_names, calorie_df)
        
        # Evaluate classification performance
        class_results = evaluate_classification(model, device, test_dataset)
        
        # Evaluate calorie estimation performance
        calorie_results = evaluate_calorie_estimation(
            class_results['predictions'],
            class_results['labels'],
            calorie_mapping,
            class_names
        )
        
        # Visualize results
        visualize_classification_results(class_results, class_names, output_dir)
        visualize_calorie_estimation(calorie_results, output_dir)
        
        # Analyze challenging cases
        analyze_challenging_cases(class_results, calorie_results, class_names, test_dataset, calorie_mapping, output_dir)
        
        # Generate summary report
        generate_summary_report(class_results, calorie_results, output_dir)
        
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        logger.info(f"Accuracy: {class_results['accuracy']:.4f}, MAE: {calorie_results['mae']:.2f} calories per 100g")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()