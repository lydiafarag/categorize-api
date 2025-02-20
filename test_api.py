import requests

API_URL = "https://categorize-api-1.onrender.com/process/"
image_path = "sept 26_LB_training_NC,PC,_60mins.jpeg"  # Replace with the actual image path

with open(image_path, "rb") as img:
    files = {"file": img}
    data = {"model_type": "multiclass"}  # Or "multiclass"

    response = requests.post(API_URL, files=files, data=data)

print(response.json())  # Should return classification results
