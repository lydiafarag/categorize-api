import base64
import shutil
import uvicorn
import cv2
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.segmentations_utils import segment_image
from api.classification_utils import classify_images
import os
import tensorflow as tf
#added this for render to be able to run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')
# Initialize FastAPI app
app = FastAPI()

# Allow requests from the React Native front-end
origins = [
    "http://localhost",  # Replace with the URL of your React Native app
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
UPLOAD_DIR = Path("api/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTED_DIR = Path("api/segmented_wells")
SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the Well Classification API!"}

@app.post("/process/")
async def process_image(file: UploadFile = File(...), model_type: str = "binary"):
    try:
        # Convert `file.filename` to a Path object BEFORE using `.stem`
        file_path = Path(file.filename)
        
        # Ensure safe file saving with `.name`
        upload_path = UPLOAD_DIR / file_path.name  
        
        # Save the uploaded image
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Create segmented folder using `.stem`
        segmented_images_folder = SEGMENTED_DIR / file_path.stem  
        segmented_images_folder.mkdir(exist_ok=True)

        # Perform segmentation
        rows, columns, radius = segment_image(upload_path, segmented_images_folder)

        # Classify segmented wells
        classification_results = classify_images(segmented_images_folder, model_type, rows, columns, radius)
        print(f"Classified {len(classification_results)}/96 wells")

        return JSONResponse(content={"results": classification_results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def create_annotated_image(original_path, results, save_path):
    """
    Annotate the original image with classification results.
    """
    image = cv2.imread(str(original_path))
    for result in results:
        x, y, r = result["circle"]
        label = result["classification"]
        color = (0, 255, 0) if "Effective" in label else (0, 0, 255)
        cv2.circle(image, (x, y), r, color, 2)
        cv2.putText(
            image,
            label,
            (x - 20, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(save_path), image)

def image_to_base64(image_path):
    """
    Convert an image to Base64 string.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)