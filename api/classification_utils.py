import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to match CNN input
    image = image.astype("float32") / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

def classify_images(segmented_images_folder, model_type, rows, columns, radius):

    # Select and load model
    if model_type == "binary":
        model_path = "api/models/well_classifier_cnn.h5"
    elif model_type == "multiclass":
        model_path = "api/models/well_classifier_cnn_4_classes.h5"
    else:
        raise Exception(f"No model of type \"{model_type}\" found.")
    
    model = load_model(model_path)
    predictions = []
    
    # Loop through each image in the folder
    for image_name in os.listdir(segmented_images_folder):
        if image_name.startswith("circle_") and image_name.endswith(('.jpg', '.jpeg', '.png')):
            try:
                # Extract row and column index from filename
                parts = image_name.replace("circle_", "").replace(".png", "").split("_")
                row_index = int(parts[0])
                column_index = int(parts[1])

                # Get corresponding x and y from ordered lists
                x = columns[column_index-1]
                y = rows[row_index-1]

                # Load the image
                image_path = os.path.join(segmented_images_folder, image_name)
                image = preprocess_image(image_path)

                # Predict the label using your model
                prediction = model.predict(image)
                predicted_label = "Effective" if np.argmax(prediction) == 1 else "Ineffective"

                # Store results
                predictions.append({
                    "x": x,
                    "y": y,
                    "r": radius,
                    "Predicted": predicted_label
                })

            # Unexpected filenames
            except (IndexError, ValueError) as e:
                print(f"Skipping file {image_name}: {e}") 

    return predictions