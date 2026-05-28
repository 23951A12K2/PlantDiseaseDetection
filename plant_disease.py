import os
import numpy as np
import datetime

# Import required TensorFlow and Keras modules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

# Dataset and model paths
DATASET_PATH = r"C:\Users\anuma\OneDrive\Documents\PlantDataset\PlantVillage"
MODEL_PATH = "plant_disease_model.keras"

# Image and training settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE = 5

# Learning rates
LR_HEAD = 1e-3
LR_FINE = 1e-5

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
    validation_split=0.2
)

# Validation image preprocessing
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# Load training dataset
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

# Load validation dataset
val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

# Total number of classes
NUM_CLASSES = train_data.num_classes

# Calculate class weights for handling class imbalance
class_counts = np.bincount(train_data.classes)
total_samples = len(train_data.classes)

class_weights = {
    i: total_samples / (NUM_CLASSES * count)
    for i, count in enumerate(class_counts)
}

# Check if trained model already exists
if os.path.exists(MODEL_PATH):

    print("Loading existing model...")
    model = load_model(MODEL_PATH)

else:

    print("Training new model...")

    # Load MobileNetV2 pretrained model
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    # Freeze base model layers initially
    base.trainable = False

    # Add custom classification layers
    x = base.output
    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Output layer
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    # Create complete model
    model = Model(inputs=base.input, outputs=output)

    # Compile model for Phase 1 training
    model.compile(
        optimizer=Adam(learning_rate=LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # TensorBoard log directory
    log_dir = os.path.join(
        "logs",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # Callbacks for Phase 1
    callbacks_phase1 = [

        # Save best model during head training
        ModelCheckpoint(
            "best_head.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        # Stop training if validation accuracy stops improving
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate automatically
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),

        # TensorBoard visualization
        TensorBoard(log_dir=log_dir)
    ]

    # Phase 1 training
    print("Phase 1: Training classification head")

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=callbacks_phase1
    )

    # Unfreeze top layers for fine-tuning
    base.trainable = True

    # Freeze all layers except top 30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False

    # Recompile model for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LR_FINE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks for Phase 2
    callbacks_phase2 = [

        # Save best final model
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        # Early stopping for fine-tuning
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate during fine-tuning
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-8,
            verbose=1
        ),

        # TensorBoard logging
        TensorBoard(log_dir=log_dir)
    ]

    # Phase 2 fine-tuning
    print("Phase 2: Fine-tuning MobileNetV2 layers")

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FINE,
        class_weight=class_weights,
        callbacks=callbacks_phase2
    )

    # Save trained model
    model.save(MODEL_PATH)
    print("Model saved successfully")