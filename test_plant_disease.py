import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, classification_report

DATASET_PATH = r"C:\Users\anuma\OneDrive\Documents\PlantDataset\PlantVillage"
MODEL_PATH   = "plant_disease_model.keras"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32

model = load_model(MODEL_PATH)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

predictions = model.predict(val_data, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_data.classes
class_names = list(val_data.class_indices.keys())

accuracy  = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average="weighted", zero_division=0)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print("\nPer-Class Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names, zero_division=0))