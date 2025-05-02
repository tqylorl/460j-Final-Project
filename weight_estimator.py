import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from food_classifier import classify_food
import json
import os
from datasets import load_dataset

class WeightEstimator(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super().__init__()
        self.base = timm.create_model(
            model_name, pretrained=False,
            num_classes=0, global_pool='avg'
        )
        nf = self.base.num_features  # 1280 for efficientnet_b0
        self.fusion = nn.Sequential(
            nn.Linear(nf + 101, 512),  # 101 food classes
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x, ft):
        feats = self.base(x)
        return self.fusion(torch.cat([feats, ft], dim=1))

class FoodWeightEstimator:
    def __init__(self, model_path='models/final_model.pth'):
        # Initialize the model
        self.model = WeightEstimator('efficientnet_b0')
        
        # Load the trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model.eval()
        
        # Load Food101 class names
        self.dataset = load_dataset("food101", split="train")
        self.food_labels = self.dataset.features['label'].names
        
        # Image preprocessing - using the same transforms as training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

        # Extended food density database (grams per cubic cm)
        self.density_db = {
            "pizza": 0.8,
            "hamburger": 0.9,
            "sushi": 1.0,
            "steak": 1.2,
            "salad": 0.3,
            "pasta": 0.7,
            "rice": 0.6,
            "bread": 0.4,
            "cake": 0.5,
            "ice_cream": 0.6,
            "soup": 1.0,
            "french_fries": 0.5,
            "chicken_curry": 0.9,
            "beef_stew": 1.1,
            "fish": 1.0,
            "sandwich": 0.6,
            "taco": 0.8,
            "burrito": 0.9,
            "apple_pie": 0.7,
            "chocolate_cake": 0.6
        }
        
        # Extended typical portion sizes (in grams)
        self.portion_sizes = {
            "pizza": (150, 350),      # Slice to medium pizza
            "hamburger": (120, 250),  # Regular to large burger
            "sushi": (30, 150),       # Single piece to roll
            "pasta": (150, 350),      # Regular to large portion
            "salad": (100, 250),      # Side to main course
            "ice_cream": (80, 200),   # Small to large serving
            "chocolate_cake": (80, 180),  # Slice sizes
            "apple_pie": (120, 200),  # Slice sizes
            "chicken_curry": (180, 400),  # Regular to large serving
            "beef_stew": (180, 400),   # Regular to large serving
            "rice": (100, 250),        # Side to main portion
            "bread": (30, 100),        # Slice to multiple slices
            "sandwich": (150, 300),    # Regular to club sandwich
            "french_fries": (80, 200), # Small to large portion
            "chicken_wings": (120, 300), # Few to many wings
            "taco": (100, 200),        # Single to large taco
            "burrito": (180, 350),     # Regular to large burrito
            "soup": (200, 400),        # Bowl sizes
            "steak": (150, 350),       # Different cut sizes
            "fish": (120, 250)         # Fillet sizes
        }
    
    def _create_food_feature(self, food_type):
        """Create one-hot encoded feature vector for food type"""
        ft = torch.zeros(101)  # 101 food classes
        try:
            food_idx = self.dataset.features['label'].str2int(food_type)
            ft[food_idx] = 1
        except ValueError:
            # If food type not found, use zeros (could be improved)
            pass
        return ft
    
    def _adjust_weight_estimate(self, food_type, raw_estimate):
        """Adjust the raw model estimate based on food type and typical portions"""
        food_type = food_type.lower().replace(" ", "_")
        
        # Get typical range for this food
        min_weight, max_weight = self.portion_sizes.get(
            food_type, (100, 300)  # Default range if food not found
        )
        
        # Get food density
        density = self.density_db.get(food_type, 1.0)
        
        # Adjust the raw estimate to be within reasonable bounds
        adjusted_weight = np.clip(raw_estimate * density, min_weight, max_weight)
        
        return adjusted_weight
    
    def estimate_weight(self, image_path):
        """
        Estimate the weight of food in the image
        Args:
            image_path: Path to the food image
        Returns:
            estimated_weight: in grams
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        # Get food type and create feature vector
        food_type = classify_food(image_path)
        food_feature = self._create_food_feature(food_type).unsqueeze(0)
        
        with torch.no_grad():
            # Get weight prediction
            output = self.model(image_tensor, food_feature)
            raw_estimate = float(output.item())
            
            # Adjust the estimate based on food type
            adjusted_weight = self._adjust_weight_estimate(food_type, raw_estimate)
            
            return max(0, adjusted_weight)  # Ensure non-negative weight
    
    def analyze_food_weight(self, image_path):
        """
        Complete analysis of food weight including classification and estimation
        """
        # First classify the food
        food_type = classify_food(image_path)
        
        # Then estimate weight
        estimated_weight = self.estimate_weight(image_path)
        
        return {
            'food_type': food_type,
            'estimated_weight_grams': round(estimated_weight, 1)
        }

def main():
    # Example usage
    estimator = FoodWeightEstimator()
    
    # Test with an image
    test_image = "path_to_test_image.jpg"
    if os.path.exists(test_image):
        result = estimator.analyze_food_weight(test_image)
        print("\nFood Weight Analysis:")
        print(f"Food Type: {result['food_type']}")
        print(f"Estimated Weight: {result['estimated_weight_grams']}g")

if __name__ == "__main__":
    main() 