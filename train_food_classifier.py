import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets import load_dataset
from transformers import AutoImageProcessor

# use gpu if we can (only works for nvidia for some reason ??? rip amd ig)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load Food-101
dataset = load_dataset("food101", split = "train[:5%]") # using small subset for testing

# load processor
processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224", use_fast=True)

# define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# define dataset wrapper
class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.labels = sorted(list(set(self.dataset['label'])))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# create pytorch datasets
train_ds = FoodDataset(dataset, transform=transform)
train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)

# load pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_classes = 101 # food 101
model.fc = nn.Linear(model.fc.in_features, num_classes)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_ld:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_ld):.4f}")

print("training completed...")

# save model
torch.save(model.state_dict(), "./models/food101_resnet18.pth")
print("model saved!")

