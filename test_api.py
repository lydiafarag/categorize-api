import requests
import base64
import os

API_URL = "https://categorize-api-1.onrender.com/process/"
image_path = os.path.join(os.path.dirname(__file__), "EC_test.jpeg")


#check if the file exists before reading it
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: The file '{image_path}' does not exist.")

#convert to base64 format
with open(image_path, "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode("utf-8")  # Proper UTF-8 encoding

is_uploaded = True  # modified for testing purposes

#payload for API request
data = {
    "image_to_base64": base64_image,  #base64 encoded image
    "model_type": "rule",  #change as needed
    "isUpload": is_uploaded  # true or false
}

# Send POST request to the API
response = requests.post(API_URL, json=data)

# Handle response
try:
    response_json = response.json()  # parse JSON resp
    print("Response JSON:", response_json)  # Display the response
except requests.exceptions.JSONDecodeError:
    print("Error decoding JSON response:", response.text)  # debugging for invalid JSON response
