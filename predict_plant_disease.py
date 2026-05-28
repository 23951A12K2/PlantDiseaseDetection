import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

MODEL_PATH   = "plant_disease_model.keras"
DATASET_PATH = r"C:\Users\anuma\OneDrive\Documents\PlantDataset\PlantVillage"
IMG_SIZE     = (224, 224)

model = load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
reference_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=False
)

class_names = list(reference_data.class_indices.keys())

def predict_disease(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    predictions = model.predict(arr, verbose=0)
    class_index = np.argmax(predictions[0])
    confidence  = predictions[0][class_index] * 100
    class_label = class_names[class_index]

    print(f"\nPredicted Disease : {class_label}")
    print(f"Confidence        : {confidence:.2f}%")

    print("\nTop 3 Predictions:")
    top3_indices = np.argsort(predictions[0])[::-1][:3]
    for i, idx in enumerate(top3_indices):
        print(f"  {i+1}. {class_names[idx]} - {predictions[0][idx] * 100:.2f}%")

    return class_label, confidence


while True:
    image_path = input("\nEnter image path (or 'q' to quit): ").strip().strip('"').strip("'")

    if image_path.lower() == 'q':
        break

    if not os.path.exists(image_path):
        print("File not found. Please enter a valid path.")
        continue

    predict_disease(image_path)