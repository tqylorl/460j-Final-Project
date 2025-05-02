import torch
import torchvision.models as models
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFile
import warnings
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import json
import os

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Food101Dataset(Dataset):
    def __init__(self, split="validation"):
        self.dataset = load_dataset('food101', split=split, trust_remote_code=True)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.length = len(self.dataset)
    
    def __len__(self):
        return self.length
    
    def safe_get_item(self, idx):
        """Safely get an item from the dataset without EXIF processing"""
        if idx >= self.length:
            warnings.warn(f"Index {idx} out of bounds for dataset of size {self.length}")
            return None, None
            
        try:
            item = self.dataset[idx]
            image = item['image']
            label = item['label']
            
            if isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image))
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            return image, label
        except Exception as e:
            warnings.warn(f"Error getting item at index {idx}: {str(e)}")
            return None, None
    
    def __getitem__(self, idx):
        image, label = self.safe_get_item(idx)
        
        if image is None:
            return torch.zeros(3, 224, 224), 0
        
        try:
            image = self.transform(image)
        except Exception as e:
            warnings.warn(f"Error transforming image at index {idx}: {str(e)}")
            return torch.zeros(3, 224, 224), label
        
        return image, label

def create_visualizations(all_labels, all_predictions, class_names, accuracy, class_correct, class_total):
    """Create and save visualizations of model performance"""
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()
    
    # Create class accuracy bar plot
    class_accuracies = []
    for i in range(len(class_names)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            class_accuracies.append((class_names[i], acc))
    
    # Sort by accuracy
    class_accuracies.sort(key=lambda x: x[1])
    
    # Plot class accuracies
    plt.figure(figsize=(15, 30))
    classes, accs = zip(*class_accuracies)
    plt.barh(range(len(classes)), accs)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Accuracy (%)')
    plt.title('Class-wise Accuracy')
    plt.tight_layout()
    plt.savefig('visualizations/class_accuracy.png')
    plt.close()
    
    # Create summary statistics
    stats = {
        'Overall Accuracy': f'{accuracy:.2f}%',
        'Top 5 Classes': [f'{name}: {acc:.2f}%' for name, acc in class_accuracies[-5:]],
        'Bottom 5 Classes': [f'{name}: {acc:.2f}%' for name, acc in class_accuracies[:5]]
    }
    
    # Save statistics to JSON file
    with open('visualizations/model_performance_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Also save as text file for easy reading
    with open('visualizations/model_performance_stats.txt', 'w') as f:
        f.write('Model Performance Statistics\n')
        f.write('=========================\n\n')
        f.write(f'Overall Accuracy: {stats["Overall Accuracy"]}\n\n')
        f.write('Top 5 Performing Classes:\n')
        for stat in stats['Top 5 Classes']:
            f.write(f'{stat}\n')
        f.write('\nBottom 5 Performing Classes:\n')
        for stat in stats['Bottom 5 Classes']:
            f.write(f'{stat}\n')

def evaluate_and_visualize(model, dataloader, class_names):
    """Evaluate model and create visualizations"""
    device = torch.device('mps' if torch.backends.mps.is_available() 
                         else 'cuda' if torch.cuda.is_available() 
                         else 'cpu')
    print(f"\nEvaluating on {device}...")
    
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    # Lists to store all predictions and labels for confusion matrix
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f'\nOverall Accuracy: {accuracy:.2f}%')
    
    # Create visualizations
    create_visualizations(all_labels, all_predictions, class_names, accuracy, class_correct, class_total)
    
    return accuracy, class_correct, class_total

def main():
    # Load the model
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 101)  # 101 food classes
    model.load_state_dict(torch.load('models/food101_resnet18.pth', map_location='cpu'))
    
    # Create dataset and dataloader
    dataset = Food101Dataset(split='validation')
    class_names = dataset.dataset.features['label'].names
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Evaluate and create visualizations
    evaluate_and_visualize(model, dataloader, class_names)

if __name__ == '__main__':
    main() 