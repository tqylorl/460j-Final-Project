#!/usr/bin/env python3
"""
calorie_estimator.py - Maps food classifications to calorie content

This script provides functionality to estimate calorie content from food classifications.
It implements both direct lookup and regression-based approaches, handles portion
estimation, and provides evaluation metrics.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
from typing import Dict, List, Tuple, Union, Optional

class CalorieEstimator:
    """Class to estimate calorie content from food classifications."""
    
    def __init__(self, calories_csv_path: str):
        """
        Initialize the CalorieEstimator.
        
        Args:
            calories_csv_path: Path to the calories.csv file
        """
        self.calories_df = self._load_calories_data(calories_csv_path)
        self.regression_model = None
        self.food_embeddings = {}  # Placeholder for embeddings from classification model
        self.calibration_factors = self._initialize_calibration_factors()
    
    def _load_calories_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and parse the calories.csv data.
        
        Args:
            csv_path: Path to the calories.csv file
            
        Returns:
            Pandas DataFrame containing the calories data
        """
        # Load the CSV data
        try:
            df = pd.read_csv(csv_path)
            
            # Convert calorie columns to numeric, handling any non-numeric values
            if 'Cals_per100grams' in df.columns:
                df['Cals_per100grams'] = pd.to_numeric(df['Cals_per100grams'], errors='coerce')
            
            if 'KJ_per100grams' in df.columns:
                df['KJ_per100grams'] = pd.to_numeric(df['KJ_per100grams'], errors='coerce')
            
            # Clean food item names for better matching
            if 'FoodItem' in df.columns:
                df['FoodItem_clean'] = df['FoodItem'].apply(self._clean_food_name)
            
            # Ensure no duplicates in food items
            if 'FoodItem_clean' in df.columns:
                df.drop_duplicates(subset=['FoodItem_clean'], inplace=True)
            
            return df
        
        except Exception as e:
            print(f"Error loading calories data: {e}")
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=['FoodCategory', 'FoodItem', 'per100grams', 
                                        'Cals_per100grams', 'KJ_per100grams', 'FoodItem_clean'])
    
    def _clean_food_name(self, food_name: str) -> str:
        """
        Clean and standardize food name for better matching.
        
        Args:
            food_name: Original food name
            
        Returns:
            Cleaned food name
        """
        if not isinstance(food_name, str):
            return ""
        
        # Convert to lowercase
        food_name = food_name.lower()
        
        # Remove special characters and extra spaces
        food_name = re.sub(r'[^\w\s]', ' ', food_name)
        food_name = re.sub(r'\s+', ' ', food_name).strip()
        
        # Remove common words that don't help with identification
        stop_words = ['and', 'with', 'the', 'or', 'in', 'on', 'of', 'a', 'an']
        for word in stop_words:
            food_name = re.sub(r'\b' + word + r'\b', '', food_name)
        
        # Remove extra spaces after removing stop words
        food_name = re.sub(r'\s+', ' ', food_name).strip()
        
        return food_name
    
    def _initialize_calibration_factors(self) -> Dict[str, float]:
        """
        Initialize calibration factors for different food categories.
        
        Returns:
            Dictionary mapping food categories to calibration factors
        """
        # Default calibration factor is 1.0 (no adjustment)
        default_factor = 1.0
        
        # Define specific calibration factors for different food categories
        # These would ideally be derived from empirical data
        calibration_factors = {
            'fruits': 0.8,           # Fruits often have more water content and may appear larger
            'vegetables': 0.7,       # Vegetables often have more water/fiber and fewer calories
            'desserts': 1.2,         # Desserts often have hidden calories (sugar, fat)
            'fried_foods': 1.3,      # Fried foods absorb oil, adding calories
            'fast_food': 1.2,        # Fast food often more calorie-dense than appearance suggests
            'soups': 0.6,            # Soups have high water content
            'salads': 0.5,           # Salads often low calorie unless dressing is heavy
            'pasta': 1.1,            # Pasta can be more calorie-dense than it appears
            'rice': 1.0,             # Rice is fairly consistent
            'meat': 1.2,             # Meat can have hidden fat
            'seafood': 0.9,          # Seafood often lower in calories than similar proteins
            'nuts_and_seeds': 1.5,   # Nuts and seeds are very calorie-dense
        }
        
        return calibration_factors
    
    def _find_best_match(self, food_classification: str) -> Tuple[str, float]:
        """
        Find the best match for a food classification in the calories database.
        
        Args:
            food_classification: The food classification to match
            
        Returns:
            Tuple containing the matched food item and match score
        """
        if not self.calories_df.shape[0] > 0:
            return None, 0.0
        
        # Clean the input food classification
        clean_classification = self._clean_food_name(food_classification)
        
        # If exact match exists, return it
        exact_match = self.calories_df[self.calories_df['FoodItem_clean'] == clean_classification]
        if not exact_match.empty:
            return exact_match.iloc[0]['FoodItem'], 1.0
        
        # Otherwise, find the best partial match
        best_match = None
        best_score = 0.0
        
        for _, row in self.calories_df.iterrows():
            food_item = row['FoodItem_clean']
            
            # Skip empty food items
            if not isinstance(food_item, str) or not food_item:
                continue
            
            # Calculate similarity score (simplified for this example)
            # In a real implementation, more sophisticated NLP techniques would be used
            score = self._calculate_similarity(clean_classification, food_item)
            
            if score > best_score:
                best_score = score
                best_match = row['FoodItem']
        
        # Only return match if score is above threshold
        if best_score >= 0.6:
            return best_match, best_score
        else:
            return None, 0.0
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple implementation based on word overlap
        # In a real system, you might use word embeddings or more sophisticated NLP
        
        if not str1 or not str2:
            return 0.0
        
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def direct_lookup(self, food_classification: str) -> Dict[str, Union[str, float, None]]:
        """
        Perform direct lookup to find calorie content for a food classification.
        
        Args:
            food_classification: The food classification to look up
            
        Returns:
            Dictionary with matched food item, calories per 100g, and match confidence
        """
        matched_item, confidence = self._find_best_match(food_classification)
        
        if matched_item is None:
            return {
                'matched_food': None,
                'calories_per_100g': None,
                'confidence': 0.0,
                'match_method': 'direct_lookup'
            }
        
        # Get the calorie content for the matched item
        matched_row = self.calories_df[self.calories_df['FoodItem'] == matched_item].iloc[0]
        calories_per_100g = matched_row['Cals_per100grams']
        
        return {
            'matched_food': matched_item,
            'calories_per_100g': calories_per_100g,
            'confidence': confidence,
            'match_method': 'direct_lookup'
        }
    
    def train_regression_model(self) -> None:
        """
        Train a regression model to estimate calories based on food features.
        
        In a real implementation, this would use embeddings or features from a 
        food classification model. For this example, we'll use food categories 
        and some engineered features as a proxy.
        """
        if not self.calories_df.shape[0] > 0:
            print("No calorie data available to train model")
            return
        
        # For simplicity, we'll use food categories as features
        # In a real implementation, you would use embeddings from a CV model
        
        # Prepare the data
        X = self.calories_df[['FoodCategory']].copy()
        y = self.calories_df['Cals_per100grams']
        
        # Handle missing values
        y = y.fillna(y.mean())
        
        # Add some engineered features (in a real implementation, these would come from the CV model)
        # Here we're just creating dummy features for demonstration
        X['has_meat'] = X['FoodCategory'].apply(
            lambda x: 1 if x and isinstance(x, str) and any(meat in x.lower() for meat in 
                                                         ['meat', 'beef', 'pork', 'chicken', 'fish']) else 0
        )
        
        X['is_dessert'] = X['FoodCategory'].apply(
            lambda x: 1 if x and isinstance(x, str) and any(dessert in x.lower() for dessert in 
                                                         ['dessert', 'cake', 'cookie', 'sweet']) else 0
        )
        
        X['is_vegetable'] = X['FoodCategory'].apply(
            lambda x: 1 if x and isinstance(x, str) and any(veg in x.lower() for veg in 
                                                         ['vegetable', 'veg', 'salad', 'greens']) else 0
        )
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['FoodCategory'])
            ],
            remainder='passthrough'
        )
        
        # Create and train the model
        self.regression_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        # Train the model
        self.regression_model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.regression_model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Regression Model Metrics:")
        print(f"MAE: {mae:.2f} calories")
        print(f"RMSE: {rmse:.2f} calories")
        print(f"R²: {r2:.2f}")
    
    def regression_estimate(self, food_classification: str, food_category: Optional[str] = None) -> Dict[str, Union[float, str, None]]:
        """
        Estimate calories using the regression model.
        
        Args:
            food_classification: The food classification
            food_category: Optional food category if known
            
        Returns:
            Dictionary with estimated calories per 100g and confidence
        """
        if self.regression_model is None:
            self.train_regression_model()
            
            # If still None after attempting to train, return error
            if self.regression_model is None:
                return {
                    'estimated_calories_per_100g': None,
                    'confidence': 0.0,
                    'match_method': 'regression'
                }
        
        # Try to find a matching category in our database if not provided
        if food_category is None:
            # Get the best match from direct lookup to find a category
            matched_item, _ = self._find_best_match(food_classification)
            
            if matched_item is not None:
                # Get the food category for the matched item
                matched_row = self.calories_df[self.calories_df['FoodItem'] == matched_item].iloc[0]
                food_category = matched_row['FoodCategory']
            else:
                # No matching category found, use a default
                food_category = "unknown"
        
        # Prepare the features
        X = pd.DataFrame({
            'FoodCategory': [food_category],
            'has_meat': [1 if food_category and isinstance(food_category, str) and 
                         any(meat in food_category.lower() for meat in 
                             ['meat', 'beef', 'pork', 'chicken', 'fish']) else 0],
            'is_dessert': [1 if food_category and isinstance(food_category, str) and 
                          any(dessert in food_category.lower() for dessert in 
                              ['dessert', 'cake', 'cookie', 'sweet']) else 0],
            'is_vegetable': [1 if food_category and isinstance(food_category, str) and 
                            any(veg in food_category.lower() for veg in 
                                ['vegetable', 'veg', 'salad', 'greens']) else 0]
        })
        
        # Predict calories
        estimated_calories = self.regression_model.predict(X)[0]
        
        # Calculate confidence (simplified)
        # In a real implementation, this would be based on prediction intervals
        confidence = 0.7  # Default medium confidence
        
        # Adjust confidence based on whether the food category was found in our database
        if food_category == "unknown":
            confidence = 0.4  # Lower confidence for unknown categories
        
        return {
            'estimated_calories_per_100g': max(0, estimated_calories),  # Ensure non-negative
            'confidence': confidence,
            'match_method': 'regression'
        }
    
    def estimate_portion_size(self, user_input: Optional[str] = None, 
                             image_area_pixels: Optional[int] = None,
                             food_classification: Optional[str] = None) -> float:
        """
        Estimate portion size in grams based on user input or image analysis.
        
        Args:
            user_input: Optional user input about portion size (e.g., "1 cup", "200g")
            image_area_pixels: Optional area of food in the image in pixels
            food_classification: Optional food classification to help with estimation
            
        Returns:
            Estimated portion size in grams
        """
        # Default portion size (100g)
        default_portion = 100.0
        
        # If user explicitly provided a weight in grams, use that
        if user_input:
            # Check for explicit gram specification
            gram_match = re.search(r'(\d+)\s*g', user_input)
            if gram_match:
                return float(gram_match.group(1))
            
            # Check for common portion descriptions
            if 'cup' in user_input.lower():
                cup_match = re.search(r'(\d*\.?\d+)\s*cup', user_input.lower())
                if cup_match:
                    cups = float(cup_match.group(1)) if cup_match.group(1) else 1.0
                    # Different foods have different densities per cup
                    if food_classification:
                        clean_food = food_classification.lower()
                        if any(grain in clean_food for grain in ['rice', 'grain', 'cereal']):
                            return cups * 180  # ~180g per cup for rice/grains
                        elif any(liquid in clean_food for liquid in ['soup', 'stew', 'milk', 'juice']):
                            return cups * 240  # ~240g per cup for liquids
                        elif any(veg in clean_food for veg in ['vegetable', 'salad']):
                            return cups * 150  # ~150g for chopped vegetables
                        else:
                            return cups * 200  # Default ~200g per cup
                    else:
                        return cups * 200  # Default ~200g per cup
            
            # Check for tablespoons
            if 'tbsp' in user_input.lower() or 'tablespoon' in user_input.lower():
                tbsp_match = re.search(r'(\d*\.?\d+)\s*(tbsp|tablespoon)', user_input.lower())
                if tbsp_match:
                    tbsp = float(tbsp_match.group(1)) if tbsp_match.group(1) else 1.0
                    return tbsp * 15  # ~15g per tablespoon
            
            # Check for teaspoons
            if 'tsp' in user_input.lower() or 'teaspoon' in user_input.lower():
                tsp_match = re.search(r'(\d*\.?\d+)\s*(tsp|teaspoon)', user_input.lower())
                if tsp_match:
                    tsp = float(tsp_match.group(1)) if tsp_match.group(1) else 1.0
                    return tsp * 5  # ~5g per teaspoon
            
            # Check for ounces
            if 'oz' in user_input.lower() or 'ounce' in user_input.lower():
                oz_match = re.search(r'(\d*\.?\d+)\s*(oz|ounce)', user_input.lower())
                if oz_match:
                    oz = float(oz_match.group(1)) if oz_match.group(1) else 1.0
                    return oz * 28.35  # 1 oz = 28.35g
        
        # If image area is provided, estimate based on typical food density
        if image_area_pixels and food_classification:
            # This would be calibrated based on a reference image with known dimensions
            # For simplicity, we're using a very basic estimation
            
            # Base conversion factor (pixels to grams)
            # This would ideally be calibrated with known reference objects
            base_factor = 0.1  # Hypothetical factor, would need real calibration
            
            # Find the most appropriate food category for density adjustment
            food_category = 'unknown'
            if food_classification:
                # Try to get food category from our database
                matched_item, _ = self._find_best_