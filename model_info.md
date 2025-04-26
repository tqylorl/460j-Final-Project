# Food Classifier Model Information

## Model Location
The trained model is stored locally at: `models/food_classifier_model.h5`

## Training Details
- Model architecture: MobileNetV2 (without pretrained weights)
- Image size: 96x96
- Dataset: Food-101 (extremely reduced - 1% of the original data)
- Training epochs: 3
- Final validation accuracy: ~1% (expected due to limited training and data)

## Class Mapping
Class mapping information has been saved to `class_mapping.json`

## Notes
The model was trained with random initialization instead of pretrained weights to avoid SSL certificate issues. For better performance, consider:
1. Training for more epochs
2. Using more training data
3. Using pretrained weights (after fixing SSL certificate issues) 