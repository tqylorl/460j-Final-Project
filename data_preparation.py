#!/usr/bin/env python3
# data_preparation.py
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from difflib import SequenceMatcher
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directories
os.makedirs('processed_data', exist_ok=True)
os.makedirs('model_artifacts', exist_ok=True)

def load_datasets():
    """
    Load both datasets: Food-101 from HuggingFace and calories.csv
    """
    logger.info("Loading Food-101 dataset from HuggingFace...")
    food101 = load_dataset("ethz/food101")
    
    logger.info("Loading calories.csv...")
    calories_df = pd.read_csv("calories.csv")
    
    return food101, calories_df

def string_similarity(a, b):
    """
    Calculate string similarity between two strings using SequenceMatcher
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def create_category_mapping(food101, calories_df):
    """
    Create mapping between Food-101 categories and calorie data
    using string similarity matching
    """
    logger.info("Creating mapping between Food-101 categories and calorie data...")
    
    # Get unique food categories from Food-101
    food101_categories = food101["train"].features["label"].names
    
    # Dictionary to store mappings
    category_mapping = {}
    
    # For visualization
    matched_scores = []
    
    # Clean and prepare calorie data
    calories_df['FoodCategory'] = calories_df['FoodCategory'].str.lower()
    calories_df['FoodItem'] = calories_df['FoodItem'].str.lower()
    calories_df['Cals_per100grams'] = pd.to_numeric(calories_df['Cals_per100grams'], errors='coerce')
    calories_df['KJ_per100grams'] = pd.to_numeric(calories_df['KJ_per100grams'], errors='coerce')
    
    # Process each Food-101 category
    for category in food101_categories:
        category_lower = category.lower()
        
        # Try direct matches first (more efficient)
        direct_category_match = calories_df[calories_df['FoodCategory'] == category_lower]
        direct_item_match = calories_df[calories_df['FoodItem'] == category_lower]
        
        if not direct_category_match.empty:
            best_row = direct_category_match.iloc[0]
            best_score = 1.0
            best_match = best_row['FoodItem']
        elif not direct_item_match.empty:
            best_row = direct_item_match.iloc[0]
            best_score = 1.0
            best_match = best_row['FoodItem']
        else:
            # No direct match, try fuzzy matching
            best_match = None
            best_score = 0
            best_row = None
            
            # Search in both FoodCategory and FoodItem columns
            for _, row in calories_df.iterrows():
                # Check FoodCategory
                category_score = string_similarity(category, row['FoodCategory'])
                # Check FoodItem
                item_score = string_similarity(category, row['FoodItem'])
                
                # Use the better score
                score = max(category_score, item_score)
                
                if score > best_score:
                    best_score = score
                    best_match = row['FoodItem']
                    best_row = row
        
        matched_scores.append(best_score)
        
        # Set a threshold for matches
        if best_score > 0.6:
            # Store calorie information
            category_mapping[category] = {
                'matched_item': best_match,
                'calories_per_100g': float(best_row['Cals_per100grams']) if not pd.isna(best_row['Cals_per100grams']) else None,
                'kj_per_100g': float(best_row['KJ_per100grams']) if not pd.isna(best_row['KJ_per100grams']) else None,
                'match_score': best_score
            }
        else:
            # Handle cases where no good match is found
            # Try to find a match by tokenizing the food name
            tokens = category_lower.split()
            token_matches = []
            
            for token in tokens:
                if len(token) > 3:  # Only consider tokens longer than 3 letters
                    token_category_matches = calories_df[calories_df['FoodCategory'].str.contains(token)]
                    token_item_matches = calories_df[calories_df['FoodItem'].str.contains(token)]
                    
                    if not token_category_matches.empty:
                        token_matches.append(token_category_matches.iloc[0])
                    elif not token_item_matches.empty:
                        token_matches.append(token_item_matches.iloc[0])
            
            if token_matches:
                # Use the average calorie value from token matches
                avg_calories = np.mean([row['Cals_per100grams'] for row in token_matches if not pd.isna(row['Cals_per100grams'])])
                avg_kj = np.mean([row['KJ_per100grams'] for row in token_matches if not pd.isna(row['KJ_per100grams'])])
                
                category_mapping[category] = {
                    'matched_item': f"Estimated from similar foods: {', '.join([row['FoodItem'] for row in token_matches[:3]])}...",
                    'calories_per_100g': float(avg_calories) if not np.isnan(avg_calories) else None,
                    'kj_per_100g': float(avg_kj) if not np.isnan(avg_kj) else None,
                    'match_score': best_score,
                    'estimated': True
                }
                logger.info(f"Used token-based estimation for '{category}' with score {best_score:.2f}")
            else:
                # Still no match, use a fallback strategy
                category_mapping[category] = {
                    'matched_item': 'Unknown',
                    'calories_per_100g': None,
                    'kj_per_100g': None,
                    'match_score': best_score
                }
                logger.warning(f"No good match found for '{category}' (best match: '{best_match}' with score {best_score:.2f})")
    
    # Visualize match scores
    plt.figure(figsize=(10, 6))
    plt.hist(matched_scores, bins=20)
    plt.title('Distribution of Match Scores')
    plt.xlabel('Match Score')
    plt.ylabel('Count')
    plt.savefig('model_artifacts/match_scores_distribution.png')
    
    # Print match statistics
    good_matches = sum(1 for score in matched_scores if score > 0.6)
    logger.info(f"Successfully matched {good_matches} out of {len(food101_categories)} categories ({good_matches/len(food101_categories)*100:.2f}%)")
    
    return category_mapping

def create_image_transforms():
    """
    Create image preprocessing and augmentation transforms
    """
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize larger than target to allow for cropping
        transforms.RandomCrop(224),     # Random crop for augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Validation/test transforms without augmentation
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform

class Food101Dataset(Dataset):
    """
    Custom dataset for Food-101 with calorie information
    """
    def __init__(self, dataset, category_mapping, split="train", transform=None):
        self.dataset = dataset[split]
        self.category_mapping = category_mapping
        self.transform = transform
        self.label_names = dataset[split].features["label"].names
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        
        # Convert to PIL Image if not already
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Get category name
        category = self.label_names[label]
        
        # Get calorie info
        calorie_info = self.category_mapping.get(category, {})
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Create sample with image, label and calorie info
        sample = {
            'image': image,
            'label': label,
            'category': category,
            'calories_per_100g': calorie_info.get('calories_per_100g', None),
            'kj_per_100g': calorie_info.get('kj_per_100g', None)
        }
        
        return sample

def save_processed_data(train_loader, val_loader, test_loader, category_mapping):
    """
    Save metadata and mapping information for future use
    """
    # Save category mapping
    with open('processed_data/category_mapping.json', 'w') as f:
        json.dump(category_mapping, f, indent=4)
    
    # Save dataset stats
    stats = {
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'num_categories': len(category_mapping)
    }
    
    with open('processed_data/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    logger.info(f"Saved processed data: {stats['train_samples']} training, {stats['val_samples']} validation, {stats['test_samples']} test samples")

def visualize_samples(train_loader, category_mapping):
    """
    Visualize a few samples with their calorie information
    """
    # Get a batch of samples
    dataiter = iter(train_loader)
    samples = next(dataiter)
    
    # Create a figure
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    # Denormalize images
    denorm = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # Display samples
    for i in range(min(8, len(samples['image']))):
        # Denormalize and convert to numpy
        img = denorm(samples['image'][i]).permute(1, 2, 0).clamp(0, 1).numpy()
        
        # Get info
        category = samples['category'][i]
        calories = samples['calories_per_100g'][i]
        
        # Create title text
        if calories is not None and not np.isnan(calories):
            title_text = f"{category}\n{calories:.0f} cal/100g"
        else:
            title_text = f"{category}\nNo calorie data"
        
        # Check if the calorie data is estimated
        mapping_info = category_mapping.get(category, {})
        if mapping_info.get('estimated', False):
            title_text += " (est.)"
        
        # Display
        axes[i].imshow(img)
        axes[i].set_title(title_text)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_artifacts/sample_visualization.png')

def main():
    """
    Main function to orchestrate the data preparation pipeline
    """
    # 1. Load datasets
    food101, calories_df = load_datasets()
    
    # 2. Create mapping between categories
    category_mapping = create_category_mapping(food101, calories_df)
    
    # 3. Create image transforms
    train_transform, eval_transform = create_image_transforms()
    
    # 4. Create custom datasets
    train_dataset = Food101Dataset(food101, category_mapping, split="train", transform=train_transform)
    
    # For validation and test, we'll split the validation set from Food-101
    # First, let's get the original validation set
    val_full = Food101Dataset(food101, category_mapping, split="validation", transform=eval_transform)
    
    # Calculate split sizes
    total_val = len(val_full)
    val_size = int(total_val * 0.6)  # 60% of original val set becomes our val set
    test_size = total_val - val_size  # 40% of original val set becomes our test set
    
    # Split the validation set
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_full, [val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 5. Create data loaders
    batch_size = 32
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # 6. Save processed data
    save_processed_data(train_loader, val_loader, test_loader, category_mapping)
    
    # 7. Visualize a few samples
    visualize_samples(train_loader, category_mapping)
    
    logger.info("Data preparation completed successfully!")

if __name__ == "__main__":
    main()