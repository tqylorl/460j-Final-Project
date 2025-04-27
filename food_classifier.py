import torch
from torchvision import transforms, models
from PIL import Image
import os

from datasets import load_dataset

def classify_food(image_path):
    # load trained model
    model_path = "./models/food101_resnet18.pth"
    model = models.resnet18(weights=None)

    NUM_CLASSES = 101 # food 101
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    # load saved weights
    model.load_state_dict(torch.load("./models/food101_resnet18.pth"))
    model.eval()

    # define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[.229, .224, .225]) # ImageNet
    ])

    # load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # predict
    with torch.no_grad():
        outputs = model(image)
        logits = outputs
        predicted_class_idx = logits.argmax(dim=1).item()

    dataset = load_dataset("food101", split="train")
    food_labels = dataset.features['label'].names

    predicted_label = food_labels[predicted_class_idx]

    #print(f"predicted food: {predicted_label}")
    return predicted_label

#path = input("enter image path: ")
#classify_food(path)
