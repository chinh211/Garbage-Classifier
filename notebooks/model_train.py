# @title
# -*- coding: utf-8 -*-
"""
Garbage Classifier - Improved Training Pipeline
"""

# 1. SETUP & IMPORTS
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

# 2. CONFIGURATION
class Config:
    # Paths
    PROJECT_PATH = '/content/drive/MyDrive/Garbage_Classifier'
    DATA_PATH = os.path.join(PROJECT_PATH, 'dataset')
    SAVED_MODEL_DIR = os.path.join(PROJECT_PATH, 'saved_models')
    LOG_DIR = os.path.join(PROJECT_PATH, 'logs')

    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0005
    EPOCHS = 50

    # Model selection: 'mobilenetv2' or 'efficientnetb0'
    BASE_MODEL_TYPE = 'mobilenetv2'

    # Training parameters
    FINE_TUNE_AT = 60  # Layer to start fine-tuning from
    FINE_TUNE_EPOCHS = 40  # Additional epochs for fine-tuning

config = Config()

# Create directories if they don't exist
os.makedirs(config.SAVED_MODEL_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)

# 3. DATA PREPARATION
def create_data_generators(config):
    """Create augmented training and validation generators"""

    # Enhanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=35,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=20.0,
        horizontal_flip=True,
        brightness_range=[0.75, 1.15],
        fill_mode='nearest',
        validation_split=0.2,
        vertical_flip=True
    )

    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        config.DATA_PATH,
        target_size=config.IMAGE_SIZE, # FIXED: Was config.image_size
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        config.DATA_PATH,
        target_size=config.IMAGE_SIZE, # FIXED: Was config.image_size
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

# Create data generators
train_gen, val_gen = create_data_generators(config)
num_classes = train_gen.num_classes

print(f"\n{'='*60}")
print(f"Dataset Information:")
print(f"{'='*60}")
print(f"Number of classes: {num_classes}")
print(f"Class indices: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"{'='*60}\n")

# 4. MODEL CREATION
def create_model(base_model_type, image_size, num_classes):
    """Creates a new model with a frozen base"""
    if base_model_type == 'mobilenetv2':
        base_model = MobileNetV2(
            input_shape=(*image_size, 3),
            include_top=False,
            weights='imagenet'
        )
    elif base_model_type == 'efficientnetb0':
        base_model = EfficientNetB0(
            input_shape=(*image_size, 3),
            include_top=False,
            weights='imagenet'
        )
    
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

model, base_model = create_model(config.BASE_MODEL_TYPE, config.IMAGE_SIZE, num_classes)

# 5. CALLBACKS
def create_callbacks(config, phase):
    """Create training callbacks for a specific phase"""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Model checkpoint
    checkpoint_path = os.path.join(
        config.SAVED_MODEL_DIR,
        f'best_model_{phase}_{timestamp}.h5'
    )
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    # TensorBoard
    tensorboard = TensorBoard(
        log_dir=os.path.join(config.LOG_DIR, f'{phase}_{timestamp}'),
        histogram_freq=1
    )

    return [checkpoint, reduce_lr, early_stop, tensorboard]

# 6. TRAINING - PHASE 1 (Feature Extraction)
print("\n" + "="*60)
print("PHASE 1: Training with frozen base model")
print("="*60 + "\n")

model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

model.summary()

history_phase1 = model.fit(
    train_gen,
    epochs=config.EPOCHS,
    validation_data=val_gen,
    callbacks=create_callbacks(config, 'phase1'),
    verbose=1
)

# 7. TRAINING - PHASE 2 (Fine-tuning)
print("\n" + "="*60)
print("PHASE 2: Fine-tuning base model")
print("="*60 + "\n")

base_model.trainable = True

for layer in base_model.layers[:config.FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

history_phase2 = model.fit(
    train_gen,
    epochs=config.FINE_TUNE_EPOCHS,
    validation_data=val_gen,
    callbacks=create_callbacks(config, 'phase2'),
    initial_epoch=len(history_phase1.history['loss']),
    verbose=1
)

# 8. SAVE FINAL MODEL
final_model_path = os.path.join(config.SAVED_MODEL_DIR, 'garbage_classifier_final.h5')
model.save(final_model_path)
print(f"\n✓ Final model saved to: {final_model_path}")

class_indices_path = os.path.join(config.SAVED_MODEL_DIR, 'class_indices.json')
with open(class_indices_path, 'w') as f:
    json.dump(train_gen.class_indices, f)
print(f"✓ Class indices saved to: {class_indices_path}")

# 9. VISUALIZATION
def plot_training_history(history1, history2):
    """Plot training and validation metrics for both phases"""

    acc = history1.history.get('accuracy', []) + history2.history.get('accuracy', [])
    val_acc = history1.history.get('val_accuracy', []) + history2.history.get('val_accuracy', [])
    loss = history1.history.get('loss', []) + history2.history.get('loss', [])
    val_loss = history1.history.get('val_loss', []) + history2.history.get('val_loss', [])

    epochs_range = range(len(acc))
    phase1_end = len(history1.history.get('accuracy', []))

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=phase1_end -1, color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=phase1_end - 1, color='r', linestyle='--', label='Fine-tuning starts')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.SAVED_MODEL_DIR, 'training_history.png'))
    plt.show()

plot_training_history(history_phase1, history_phase2)

# 10. EVALUATION
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60 + "\n")

val_loss, val_accuracy, val_top3_accuracy = model.evaluate(val_gen, verbose=1)

print(f"\nValidation Results:")
print(f"  Loss: {val_loss:.4f}")
print(f"  Accuracy: {val_accuracy*100:.2f}%")
print(f"  Top-3 Accuracy: {val_top3_accuracy*100:.2f}%")