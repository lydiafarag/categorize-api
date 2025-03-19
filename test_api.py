import requests
import base64
import os

API_URL = "https://categorize-api-1.onrender.com/process/"
image_path = "sept_26_LB_training_NC_PC_60mins.jpeg"  # Replace with your actual image path

# Ensure the file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: The file '{image_path}' does not exist.")

# Read image and convert to Base64
with open(image_path, "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode("utf-8")  # Proper UTF-8 encoding

# Print Base64 length for debugging (to check if image is encoded correctly)
print("Base64 Length:", len(base64_image))

# Ensure isupload is a boolean (not a string)
# If the filename contains "upload", assume it's an uploaded image; otherwise, assume captured
is_uploaded = "upload" in image_path.lower()  # Modify as needed for testing

# Prepare the properly formatted JSON payload
data = {
    "image_to_base64": base64_image,  # Ensure this is a proper Base64 string
    "model_type": "multiclass",  # Change to "binary" if needed
    "isupload": is_uploaded  # Ensure this is a boolean, not a string
}

# Print the payload for debugging (without printing the Base64 data)
print("Payload being sent:", {k: v if k != "image_to_base64" else "Base64 Data Hidden" for k, v in data.items()})

# Make the API request
response = requests.post(API_URL, json=data)

# Handle response
try:
    response_json = response.json()  # Attempt to parse JSON response
    print("Response JSON:", response_json)  # Display the response
except requests.exceptions.JSONDecodeError:
    print("Error decoding JSON response:", response.text)  # Print error response if JSON is invalid
