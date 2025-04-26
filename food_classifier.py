#!/usr/bin/env python3
# food_classifier.py
# Purpose: Define and train a food classification model using the Food-101 dataset

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2  # Changed from EfficientNetB0 to MobileNetV2 (faster)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import ssl

# Try to handle SSL certificate issues
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    # If certifi is not available, disable verification (less secure)
    if hasattr(ssl, '_create_unverified_context'):
        ssl._create_default_https_context = ssl._create_unverified_context

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("Starting with TensorFlow version:", tf.__version__)

# Set TensorFlow to use CPU only as per requirements
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("CUDA_VISIBLE_DEVICES set to: ", os.environ["CUDA_VISIBLE_DEVICES"])

def load_and_preprocess_data():
    """
    Load the Food-101 dataset from HuggingFace and preprocess it
    """
    print("Loading Food-101 dataset...")
    dataset = load_dataset("ethz/food101")
    
    # Extract training and validation sets
    train_ds = dataset["train"]
    val_ds = dataset["validation"] if "validation" in dataset else dataset["test"]
    
    # EXTREME SPEED OPTIMIZATION: Use only 1% of the data
    train_ds = train_ds.select(range(0, len(train_ds), 100))  # Take every 100th example
    val_ds = val_ds.select(range(0, len(val_ds), 50))        # Take every 50th example
    
    print(f"Using extremely reduced dataset: {len(train_ds)} training samples, {len(val_ds)} validation samples")
    
    # Get the class names
    class_names = train_ds.features["label"].names
    num_classes = len(class_names)
    
    # Define image size - reduced from 224x224 to 160x160 for faster processing
    img_size = (96, 96)  # Even smaller images
    print(f"Using image size: {img_size}")
    
    # Define preprocessing function for images
    def preprocess_image(example):
        # Resize and normalize images
        image = example["image"]
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return {"image": image, "label": example["label"]}
    
    print("Starting image preprocessing...")
    # Apply preprocessing
    train_ds = train_ds.map(preprocess_image)
    val_ds = val_ds.map(preprocess_image)
    print("Image preprocessing completed")
    
    # Convert to TensorFlow datasets and batch
    # SPEED OPTIMIZATION: Increase batch size if your GPU has enough memory
    batch_size = 128  # Even larger batch size
    print(f"Using batch size: {batch_size}")
    
    def create_tf_dataset(hf_dataset):
        print(f"Creating TensorFlow dataset from {len(hf_dataset)} examples...")
        # Extract images and labels
        images = np.array([example["image"] for example in hf_dataset])
        print(f"Images array shape: {images.shape}")
        labels = np.array([example["label"] for example in hf_dataset])
        print(f"Labels array shape: {labels.shape}")
        
        # Convert labels to one-hot encoding
        print("Converting to one-hot encoding...")
        labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        print(f"One-hot labels shape: {labels_one_hot.shape}")
        
        # Create and return TensorFlow dataset
        print("Creating final TensorFlow dataset...")
        return tf.data.Dataset.from_tensor_slices((images, labels_one_hot)).batch(batch_size)
    
    # Create TensorFlow datasets
    print("Processing training dataset...")
    tf_train_ds = create_tf_dataset(train_ds)
    print("Processing validation dataset...")
    tf_val_ds = create_tf_dataset(val_ds)
    
    print(f"Dataset loaded with {num_classes} food categories")
    return tf_train_ds, tf_val_ds, class_names

def create_model(num_classes):
    """
    Create a CNN model for food classification using MobileNetV2 with transfer learning
    """
    print("Creating model with MobileNetV2 architecture...")
    
    # Load MobileNetV2 model without pretrained weights to avoid download issues
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(96, 96, 3))
    print(f"Base model loaded with {len(base_model.layers)} layers")
    
    # Freeze the base model layers for initial training
    base_model.trainable = False
    print("Base model layers frozen")
    
    # Add custom classification head - simplified for faster training
    x = base_model.output
    print(f"Base model output shape: {base_model.output.shape}")
    x = GlobalAveragePooling2D()(x)
    print(f"After GlobalAveragePooling: {x.shape}")
    x = Dropout(0.5)(x)  # Add dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)
    print(f"Final output layer shape: {predictions.shape}")
    
    # Create the complete model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled successfully")
    
    # Print model summary
    model.summary()
    
    print(f"Model created with {len(model.layers)} layers")
    return model, base_model

def setup_callbacks():
    """
    Set up callbacks for model training (early stopping, checkpointing, learning rate reduction)
    """
    print("Setting up training callbacks...")
    # Create directory for model checkpoints
    os.makedirs('models', exist_ok=True)
    
    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        'models/food_classifier_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,  # Even more aggressive early stopping
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,  # Even more aggressive learning rate reduction
        min_lr=1e-6,
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]

def train_model(model, train_ds, val_ds, callbacks, epochs=3):  # Reduced to just 3 epochs
    """
    Train the model with the provided datasets
    """
    print(f"Training model for {epochs} epochs...")
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2  # More verbose output
    )
    
    print("Initial training completed")
    return history

def save_class_mapping(class_names):
    """
    Save class names to a file for later use in inference
    """
    print("Saving class mapping...")
    class_mapping = {i: name for i, name in enumerate(class_names)}
    with open('class_mapping.json', 'w') as f:
        import json
        json.dump(class_mapping, f)
    print("Class mapping saved to 'class_mapping.json'")

def main():
    """
    Main function to run the training pipeline
    """
    try:
        print("\n=== STARTING FOOD CLASSIFIER TRAINING ===\n")
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        
        # Load and preprocess the data
        print("\n=== DATA LOADING AND PREPROCESSING ===\n")
        train_ds, val_ds, class_names = load_and_preprocess_data()
        
        # Create the model
        print("\n=== MODEL CREATION ===\n")
        model, base_model = create_model(len(class_names))
        
        # Set up callbacks
        print("\n=== SETTING UP CALLBACKS ===\n")
        callbacks = setup_callbacks()
        
        # Train the model
        print("\n=== STARTING INITIAL TRAINING ===\n")
        history = train_model(model, train_ds, val_ds, callbacks)
        
        # Save the model and class mapping
        print("\n=== SAVING MODEL AND CLASS MAPPING ===\n")
        model.save('models/food_classifier_model.h5')
        save_class_mapping(class_names)
        
        print("\n=== TRAINING COMPLETED SUCCESSFULLY! ===\n")
        
    except Exception as e:
        print(f"\n=== ERROR DURING TRAINING: {str(e)} ===\n")
        import traceback
        traceback.print_exc()
        # Still try to save the model if possible
        try:
            if 'model' in locals() and model is not None:
                print("Attempting to save model despite error...")
                model.save('models/food_classifier_model_partial.h5')
                print("Partial model saved")
            if 'class_names' in locals() and class_names is not None:
                save_class_mapping(class_names)
        except:
            print("Could not save partial results")

if __name__ == "__main__":
    main()