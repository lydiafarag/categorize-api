import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

ROW_LABELS = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # resize to mach the input size of the model
    image = image.astype("float32") / 255.0  # normalizatio
    return np.expand_dims(image, axis=0)  # add batch dimension

def classify_images(segmented_images_folder, model_type, rows, columns, radius):

  
    if model_type == "binary":
        model_path = "api/models/well_classifier_cnn.h5"
    elif model_type == "multiclass":
        model_path = "api/models/well_classifier_cnn_3_classes_v3.h5"
    else:
        raise Exception(f"No model of type \"{model_type}\" found.")
    
    model = load_model(model_path)
    predictions = []
    MULTICLASS_LABELS = ["Ineffective", "Somewhat Effective", "Effective"]
    
   
    for image_name in os.listdir(segmented_images_folder):
        if image_name.startswith("circle_") and image_name.endswith(('.jpg', '.jpeg', '.png')):
            try:
           
                parts = image_name.replace("circle_", "").replace(".png", "").split("_")
                row_index = int(parts[0])
                column_index = int(parts[1])

                
                x = columns[column_index-1]
                y = rows[row_index-1]

                
                row = ROW_LABELS[row_index]
                column = column_index

              
                image_path = os.path.join(segmented_images_folder, image_name)
                image = preprocess_image(image_path)

               
                prediction = model.predict(image)
                predicted_index = np.argmax(prediction)
                if model_type == "binary":
                    predicted_label = "Effective" if predicted_index == 1 else "Ineffective"
                else:  # Multiclass model
                    predicted_label = MULTICLASS_LABELS[predicted_index]  # Map to four-class labels
                # Store results
                predictions.append({
                    "x": x,
                    "y": y,
                    "r": radius,
                    "row": row,
                    "column": column,
                    "Predicted": predicted_label
                })

           
            except (IndexError, ValueError) as e:
                print(f"Skipping file {image_name}: {e}") 

    return predictions