#!/usr/bin/env python3
# food_classifier.py
# Purpose: Define and train a food classification model using the Food-101 dataset

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_and_preprocess_data():
    """
    Load the Food-101 dataset from HuggingFace and preprocess it
    """
    print("Loading Food-101 dataset...")
    dataset = load_dataset("ethz/food101")
    
    # Extract training and validation sets
    print("extracting training and validation sets")
    train_ds = dataset["train"]
    val_ds = dataset["validation"] if "validation" in dataset else dataset["test"]
    
    # Get the class names
    class_names = train_ds.features["label"].names
    num_classes = len(class_names)
    
    # Define image size for EfficientNetB0
    img_size = (224, 224)
    
    # Define preprocessing function for images
    def preprocess_image(example):
        # Resize and normalize images
        image = example["image"]
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return {"image": image, "label": example["label"]}
    
    # Apply preprocessing
    print("preprocessing images")
    train_ds = train_ds.map(preprocess_image)
    val_ds = val_ds.map(preprocess_image)
    
    # Convert to TensorFlow datasets and batch
    batch_size = 32
    
    def create_tf_dataset(hf_dataset):
        # Extract images and labels
        images = np.array([example["image"] for example in hf_dataset])
        labels = np.array([example["label"] for example in hf_dataset])
        
        # Convert labels to one-hot encoding
        labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        
        # Create and return TensorFlow dataset
        return tf.data.Dataset.from_tensor_slices((images, labels_one_hot)).batch(batch_size)
    
    # Create TensorFlow datasets
    print("creating tensorflow datasets")
    tf_train_ds = create_tf_dataset(train_ds)
    tf_val_ds = create_tf_dataset(val_ds)
    
    print(f"Dataset loaded with {num_classes} food categories")
    return tf_train_ds, tf_val_ds, class_names

def create_model(num_classes):
    """
    Create a CNN model for food classification using EfficientNetB0 with transfer learning
    """
    print("Creating model with EfficientNetB0 architecture...")
    
    # Load pre-trained EfficientNetB0 model without the top classification layer
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers for initial training
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the complete model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model created with {len(model.layers)} layers")
    return model, base_model

def setup_callbacks():
    """
    Set up callbacks for model training (early stopping, checkpointing, learning rate reduction)
    """
    # Create directory for model checkpoints
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        'model_checkpoints/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]

def train_model(model, train_ds, val_ds, callbacks, epochs=30):
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
        verbose=1
    )
    
    return history

def visualize_training(history, title_suffix=""):
    """
    Visualize the training progress (accuracy and loss curves)
    """
    print("Visualizing training progress...")
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'Model Accuracy {title_suffix}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot training and validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'Model Loss {title_suffix}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_progress{title_suffix.replace(" ", "_")}.png')
    plt.close()

def visualize_predictions(model, val_ds, class_names, num_samples=10):
    """
    Visualize sample predictions from the validation set
    """
    print("Visualizing sample predictions...")
    
    # Get a batch of validation data
    for images, labels in val_ds.take(1):
        # Make predictions
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        
        # Plot the images with predictions
        plt.figure(figsize=(20, 10))
        for i in range(min(num_samples, len(images))):
            plt.subplot(2, 5, i+1)
            plt.imshow(images[i])
            correct = predicted_classes[i] == true_classes[i]
            color = "green" if correct else "red"
            plt.title(f"True: {class_names[true_classes[i]]}\nPred: {class_names[predicted_classes[i]]}", 
                     color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png')
        plt.close()

def evaluate_model(model, val_ds, class_names):
    """
    Evaluate the model on the validation set and display metrics
    """
    print("Evaluating model performance...")
    
    # Evaluate the model
    loss, accuracy = model.evaluate(val_ds, verbose=1)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Get predictions for validation data
    all_true_classes = []
    all_pred_classes = []
    
    for images, labels in val_ds:
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        
        all_true_classes.extend(true_classes)
        all_pred_classes.extend(predicted_classes)
    
    # Generate classification report
    print("\nClassification Report:")
    report = classification_report(all_true_classes, all_pred_classes, 
                                  target_names=class_names, output_dict=True)
    
    # Display top and bottom performing classes
    report_df = pd.DataFrame(report).transpose()
    top_classes = report_df.sort_values(by='f1-score', ascending=False).head(10)
    bottom_classes = report_df.sort_values(by='f1-score', ascending=True).head(10)
    
    print("\nTop 10 performing classes:")
    print(top_classes)
    
    print("\nBottom 10 performing classes:")
    print(bottom_classes)
    
    # Generate confusion matrix visualization for top classes
    plt.figure(figsize=(12, 12))
    cm = confusion_matrix(all_true_classes, all_pred_classes)
    
    # Show confusion matrix for top 15 classes
    top_indices = np.argsort(np.diag(cm))[-15:]
    cm_top = cm[top_indices][:, top_indices]
    top_class_names = [class_names[i] for i in top_indices]
    
    sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_class_names, yticklabels=top_class_names)
    plt.title('Confusion Matrix (Top 15 Classes)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def fine_tune_model(model, base_model, train_ds, val_ds, callbacks):
    """
    Fine-tune the model by unfreezing some layers of the base model
    """
    print("Fine-tuning the model...")
    
    # Unfreeze some layers of the base model
    # First, unfreeze all layers
    base_model.trainable = True
    
    # Then, freeze the bottom layers (keeping 70% frozen, 30% trainable)
    for layer in base_model.layers[:int(len(base_model.layers) * 0.7)]:
        layer.trainable = False
    
    # Count trainable and non-trainable parameters
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    
    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,  # Fewer epochs for fine-tuning
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_class_mapping(class_names):
    """
    Save class names to a file for later use in inference
    """
    class_mapping = {i: name for i, name in enumerate(class_names)}
    with open('class_mapping.json', 'w') as f:
        import json
        json.dump(class_mapping, f)
    print("Class mapping saved to 'class_mapping.json'")

def main():
    """
    Main function to execute the training pipeline
    """
    # Load and preprocess data
    train_ds, val_ds, class_names = load_and_preprocess_data()
    
    # Create model
    model, base_model = create_model(len(class_names))
    model.summary()
    
    # Setup callbacks
    callbacks = setup_callbacks()
    
    # Initial training phase
    print("\n===== Starting Initial Training Phase =====")
    history = train_model(model, train_ds, val_ds, callbacks, epochs=20)
    
    # Visualize training progress
    visualize_training(history, title_suffix="(Initial Training)")
    
    # Fine-tuning phase
    print("\n===== Starting Fine-Tuning Phase =====")
    fine_tune_history = fine_tune_model(model, base_model, train_ds, val_ds, callbacks)
    
    # Visualize fine-tuning progress
    visualize_training(fine_tune_history, title_suffix="(Fine-Tuning)")
    
    # Evaluate model
    evaluate_model(model, val_ds, class_names)
    
    # Visualize sample predictions
    visualize_predictions(model, val_ds, class_names)
    
    # Save the final model
    model.save('food_classifier_model.h5')
    print("Model saved as 'food_classifier_model.h5'")
    
    # Save class mapping
    save_class_mapping(class_names)
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()
