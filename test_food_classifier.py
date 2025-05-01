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
        # Store the actual length of the dataset
        self.length = len(self.dataset)
    
    def __len__(self):
        return self.length
    
    def safe_get_item(self, idx):
        """Safely get an item from the dataset without EXIF processing"""
        if idx >= self.length:
            warnings.warn(f"Index {idx} out of bounds for dataset of size {self.length}")
            return None, None
            
        try:
            # Get raw item without EXIF processing
            item = self.dataset[idx]  # Changed from _data[idx] to direct indexing
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
        # Get image and label safely
        image, label = self.safe_get_item(idx)
        
        # If we failed to get the item, return a blank image and 0 label
        if image is None:
            return torch.zeros(3, 224, 224), 0
        
        # Apply transforms
        try:
            image = self.transform(image)
        except Exception as e:
            warnings.warn(f"Error transforming image at index {idx}: {str(e)}")
            return torch.zeros(3, 224, 224), label
        
        return image, label

def create_model():
    """Create and load the trained model"""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 101)  # 101 food classes
    model.load_state_dict(torch.load('models/food101_resnet18.pth', map_location='cpu'))
    return model

def create_test_dataloader(batch_size=32):
    """Create test dataloader from Food101 dataset"""
    print("Loading Food101 validation dataset...")
    dataset = Food101Dataset(split='validation')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing on MacOS
        pin_memory=False  # Disable pin_memory since it's not supported on MPS
    )
    
    # Get class names
    original_dataset = load_dataset('food101', split='validation')
    class_names = original_dataset.features['label'].names
    
    return dataloader, class_names

def evaluate(model, dataloader, class_names):
    """Evaluate model on test set"""
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
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
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
    
    # Print per-class accuracy
    print('\nPer-class Accuracy:')
    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]}: {class_acc:.2f}%')
    
    return accuracy, class_correct, class_total

def main():
    # Create model and dataloader
    model = create_model()
    dataloader, class_names = create_test_dataloader()
    
    # Evaluate
    accuracy, class_correct, class_total = evaluate(model, dataloader, class_names)
    
    # Save results
    results = {
        'overall_accuracy': accuracy,
        'per_class_accuracy': {
            class_names[i]: (100 * class_correct[i] / class_total[i]) 
            for i in range(len(class_names)) if class_total[i] > 0
        }
    }
    
    # Print top-5 and bottom-5 performing classes
    per_class_acc = [(name, acc) for name, acc in results['per_class_accuracy'].items()]
    per_class_acc.sort(key=lambda x: x[1], reverse=True)
    
    print('\nTop 5 performing classes:')
    for name, acc in per_class_acc[:5]:
        print(f'{name}: {acc:.2f}%')
    
    print('\nBottom 5 performing classes:')
    for name, acc in per_class_acc[-5:]:
        print(f'{name}: {acc:.2f}%')

if __name__ == '__main__':
    main() 