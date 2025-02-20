import requests
import base64

API_URL = "https://categorize-api-1.onrender.com/process/"
image_path = "sept 26_LB_training_NC,PC,_60mins.jpeg"  # Replace with the actual image path

# Convert image to Base64
with open(image_path, "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode("utf-8")

# Prepare the payload
data = {
    "image_to_base64": base64_image,  # Send image as Base64 string
    "model_type": "multiclass"  # Change to "binary" if needed
}

# Make the request
response = requests.post(API_URL, json=data)  # Use `json=` instead of `data=`

# Print the response
try:
    print(response.json())  # Should return classification results
except requests.exceptions.JSONDecodeError:
    print("Error decoding JSON response:", response.text)
