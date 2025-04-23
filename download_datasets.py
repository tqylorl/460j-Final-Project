#!/usr/bin/env python3
"""
Script to download and prepare datasets for the Food Image Calorie Estimator project.
"""
import os
import pandas as pd
from datasets import load_dataset
from config   import DATA_DIR, CALORIES_CSV, FOOD101_DATASET
import argparse

def download_food101():
    """Download Food-101 dataset from HuggingFace"""
    print("Downloading Food-101 dataset (this may take some time)...")
    dataset = load_dataset("ethz/food101")
    print(f"Downloaded Food-101 dataset with {len(dataset['train'])} training and {len(dataset['validation'])} validation samples")
    return dataset

def prepare_calorie_data(sample_csv_path=None):
    """Prepare the calories dataset"""
    if sample_csv_path and os.path.exists(sample_csv_path):
        print(f"Using existing calories data from {sample_csv_path}")
        df = pd.read_csv(sample_csv_path)
    else:
        # Create a sample calories dataset with common foods
        # In a real project, you would use a more comprehensive dataset
        print("Creating sample calories dataset...")
        data = {
            'FoodCategory': ['Fruits', 'Fruits', 'Vegetables', 'Vegetables', 'Grains', 
                            'Dairy', 'Protein', 'Desserts', 'Fast Food', 'Fast Food'],
            'FoodItem': ['apple', 'banana', 'carrot', 'broccoli', 'rice', 
                        'cheese', 'chicken', 'ice cream', 'pizza', 'hamburger'],
            'per100grams': ['100g', '100g', '100g', '100g', '100g', 
                            '100g', '100g', '100g', '100g', '100g'],
            'Cals_per100grams': [52, 89, 41, 34, 130, 
                                402, 165, 207, 266, 254],
            'KJ_per100grams': [218, 371, 173, 141, 544, 
                            1679, 690, 866, 1113, 1062]
        }
        df = pd.DataFrame(data)
        
        # Create a more extensive dataset by adding common food items from Food-101
        # This would be expanded in a real project
        food101 = download_food101()
        food_classes = food101['train'].features['label'].names
        
        # Add sample calorie values for Food-101 classes (these would be accurate in a real project)
        extended_data = []
        for food in food_classes:
            # Simple mapping for demonstration purposes
            if 'salad' in food:
                cal = 50
                category = 'Vegetables'
            elif 'soup' in food:
                cal = 80
                category = 'Soups'
            elif 'cake' in food or 'ice_cream' in food or 'chocolate' in food:
                cal = 300
                category = 'Desserts'
            elif 'pizza' in food or 'burger' in food or 'fries' in food:
                cal = 250
                category = 'Fast Food'
            elif 'steak' in food or 'chicken' in food or 'pork' in food or 'meat' in food:
                cal = 200
                category = 'Protein'
            elif 'pasta' in food or 'rice' in food:
                cal = 150
                category = 'Grains'
            elif 'fruit' in food or 'apple' in food or 'banana' in food:
                cal = 70
                category = 'Fruits'
            else:
                cal = 150
                category = 'Other'
            
            extended_data.append({
                'FoodCategory': category,
                'FoodItem': food.replace('_', ' '),
                'per100grams': '100g',
                'Cals_per100grams': cal,
                'KJ_per100grams': cal * 4.184
            })
        
        extended_df = pd.DataFrame(extended_data)
        df = pd.concat([df, extended_df]).drop_duplicates(subset=['FoodItem'])
    
    # Save the dataset
    os.makedirs('data', exist_ok=True)
    # Save directly to calories.csv in the current directory
    output_path = CALORIES_CSV
    df.to_csv(output_path, index=False)
    print(f"Saved calories dataset to {output_path} with {len(df)} entries")
    return df

def main():
    parser = argparse.ArgumentParser(description='Download and prepare datasets for food calorie estimation')
    parser.add_argument('--calories_csv', type=str, default=None, help='Path to existing calories.csv (if available)')
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Download and prepare datasets
    download_food101()
    prepare_calorie_data(args.calories_csv)
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()