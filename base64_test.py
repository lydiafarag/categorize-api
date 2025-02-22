import base64

image_path = "sept 26_LB_training_NC,PC,_60mins.jpeg"
output_path = "reconstructed_image.jpeg"  # Save the decoded image here

# Read and encode image
with open(image_path, "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode("utf-8")

# Decode Base64 back into an image
decoded_image = base64.b64decode(base64_image)

# Save the decoded image
with open(output_path, "wb") as img_file:
    img_file.write(decoded_image)

print(f"Reconstructed image saved as {output_path}")