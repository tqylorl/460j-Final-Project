'''train_weight.py - Windows-compatible training script

Prerequisites (run once):
    pip install torch torchvision timm datasets pillow tqdm

Usage:
    python train_weight.py
'''

import os
import random
from multiprocessing import freeze_support
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from functools import partial
import shutil
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global dataset cache
_dataset_cache = {}

def get_dataset(split):
    """Get dataset from cache or load it if not cached"""
    if split not in _dataset_cache:
        try:
            logger.info(f"Loading {split} dataset...")
            _dataset_cache[split] = load_dataset("food101", split=split)
            logger.info(f"Successfully loaded {split} dataset with {len(_dataset_cache[split])} samples")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    return _dataset_cache[split]

# Make .getexif() safe on Windows
_orig_getexif = Image.Image.getexif
def _safe_getexif(self):
    try:
        return _orig_getexif(self)
    except Exception:
        return {}
Image.Image.getexif = _safe_getexif

# Custom collate function to handle errors
def collate_fn(batch):
    # Filter out None values (failed samples)
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None
    
    # Stack the tensors
    imgs = torch.stack([item[0] for item in batch])
    fts = torch.stack([item[1] for item in batch])
    wts = torch.stack([item[2] for item in batch])
    return imgs, fts, wts

# Top-level function for RGB conversion (must be picklable)
def to_rgb(img):
    try:
        return img.convert("RGB")
    except Exception as e:
        logger.error(f"Error converting image to RGB: {e}")
        return None

# Import HF datasets
try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: Please install the 'datasets' library: pip install datasets")
    exit(1)

import timm
from tqdm import tqdm

# Where to save your models
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

# Training configuration
config = {
    'model_name':  'efficientnet_b0',  # backbone
    'batch_size':  32,                 # increased batch size for 1080Ti
    'grad_accum_steps': 2,            # reduced accumulation since we can handle larger batches
    'epochs':      30,
    'lr':          1e-4,
    'image_size':  224,
    'num_workers': 4,                  # increased workers for 32GB RAM
    'pin_memory':  True,
    'prefetch_factor': 3,             # increased prefetch for better throughput
    'gpu_memory_fraction': 0.9        # use 90% of GPU memory
}

# Dataset definition
class FoodWeightDataset(Dataset):
    def __init__(self, split="train", image_size=224):
        self.dataset = get_dataset(split)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])
        # densities and portion sizes as before
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
            "ice_cream": 0.6
        }
        self.portion_sizes = {
            "pizza": (200, 400),
            "hamburger": (150, 300),
            "sushi": (100, 250),
            "steak": (150, 350),
            "salad": (100, 200),
            "pasta": (200, 400),
            "rice": (150, 300),
            "bread": (50, 150),
            "cake": (100, 250),
            "ice_cream": (100, 200)
        }

    def __len__(self):
        return len(self.dataset)

    def _estimate_weight(self, food_name):
        key = food_name.replace(" ", "_").lower()
        mn, mx = self.portion_sizes.get(key, (100,300))
        base = random.uniform(mn, mx)
        density = self.density_db.get(key, 1.0)
        return base * density * random.uniform(0.8, 1.2)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            img = item['image']
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            
            # Apply transformations
            img = self.transform(img)
            if img is None:
                logger.warning(f"Failed to transform image at index {idx}")
                return None

            label = self.dataset.features['label'].names[item['label']]
            weight = self._estimate_weight(label)

            food_idx = self.dataset.features['label'].str2int(label)
            ft = torch.zeros(len(self.dataset.features['label'].names))
            ft[food_idx] = 1

            return img, ft, torch.tensor([weight], dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error processing sample at index {idx}: {e}")
            return None

# Model definition
class WeightEstimator(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super().__init__()
        self.base = timm.create_model(
            model_name, pretrained=True,
            num_classes=0, global_pool='avg'
        )
        nf = self.base.num_features
        self.fusion = nn.Sequential(
            nn.Linear(nf + 101, 512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x, ft):
        feats = self.base(x)
        return self.fusion(torch.cat([feats, ft], dim=1))

# Training & validation functions
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, (imgs, fts, wts) in enumerate(tqdm(loader, desc="Training")):
        imgs, fts, wts = imgs.to(device), fts.to(device), wts.to(device)
        
        # Use automatic mixed precision
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.autocast('cpu'):
            preds = model(imgs, fts)
            loss = criterion(preds, wts)
            loss = loss / config['grad_accum_steps']
        
        # Accumulate gradients
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (batch_idx + 1) % config['grad_accum_steps'] == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_loss += loss.item() * config['grad_accum_steps']
        
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, fts, wts in tqdm(loader, desc="Validating"):
            imgs, fts, wts = imgs.to(device), fts.to(device), wts.to(device)
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.autocast('cpu'):
                preds = model(imgs, fts)
                loss = criterion(preds, wts)
            total_loss += loss.item()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    return total_loss / len(loader)

def save_checkpoint(model, optimizer, epoch, loss, is_best=False, filename=None):
    """Save model checkpoint with proper error handling"""
    try:
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint_path = os.path.join(models_dir, filename)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # If this is the best model, save a copy
        if is_best:
            best_path = os.path.join(models_dir, "best_model.pth")
            shutil.copy2(checkpoint_path, best_path)
            logger.info(f"Saved best model to {best_path}")
            
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        raise

# Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    # Set memory growth for 1080Ti
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set GPU memory fraction
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    logger.info(f"Models will be saved to: {models_dir}")
    
    # Verify models directory is writable
    try:
        test_file = os.path.join(models_dir, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logger.info("Models directory is writable")
    except Exception as e:
        logger.error(f"Models directory is not writable: {e}")
        raise

    logger.info("Loading datasets...")
    
    train_ds = FoodWeightDataset("train", config['image_size'])
    val_ds   = FoodWeightDataset("validation", config['image_size'])

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        collate_fn=collate_fn,
        persistent_workers=True if config['num_workers'] > 0 else False,
        prefetch_factor=config['prefetch_factor'] if config['num_workers'] > 0 else None,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        collate_fn=collate_fn,
        persistent_workers=True if config['num_workers'] > 0 else False,
        prefetch_factor=config['prefetch_factor'] if config['num_workers'] > 0 else None,
        drop_last=True
    )

    logger.info("Creating model...")
    model = WeightEstimator(config['model_name']).to(device)
    
    # Configure mixed precision training properly
    if torch.cuda.is_available():
        # Use automatic mixed precision instead of manual half precision
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = torch.autocast('cpu')
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=0.01,
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    logger.info("Starting training...")
    best_val = float('inf')
    for epoch in range(config['epochs']):
        logger.info(f"\n=== Epoch {epoch+1}/{config['epochs']} ===")
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(f"Train: {tr_loss:.4f} — Val: {val_loss:.4f}")
        
        # Save checkpoint every epoch
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            is_best=(val_loss < best_val),
            filename=f"checkpoint_epoch_{epoch+1}.pth"
        )
        
        if val_loss < best_val:
            best_val = val_loss
            logger.info(f"New best validation loss: {val_loss:.4f}")
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    save_checkpoint(
        model, optimizer, config['epochs'], best_val,
        filename="final_model.pth"
    )
    logger.info(f"Training complete! Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    freeze_support()
    main()
