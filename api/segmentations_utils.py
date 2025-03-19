import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
from scipy.cluster.hierarchy import linkage, fcluster

# Register HEIF opener for PIL to handle HEIC files
register_heif_opener()

ROWS = 8
COLUMNS = 12

# Remove outliers from rows and columns using IQR
def remove_outliers(data, m=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - m * IQR
    upper_bound = Q3 + m * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

def refine_grid(circles):
    circles = np.uint16(np.around(circles))[:, :]

    # Compute average radius
    radii = [circle[2] for circle in circles]
    average_radius = 0.8 * sum(radii) / len(radii) if radii else 0

    # Cluster to find 8 rows
    y_coords = circles[:, 1].reshape(-1, 1)
    Z_y = linkage(y_coords, method='ward')  
    row_labels = fcluster(Z_y, ROWS, criterion='maxclust')

    # Compute row averages
    row_averages = []
    for label in sorted(np.unique(row_labels)):
        row_circles = circles[row_labels == label]
        
        # Remove outliers from average calculation
        filtered_rows = remove_outliers(row_circles[:, 1])
        if len(filtered_rows) > 0:
            avg_y = int(np.mean(filtered_rows))
            row_averages.append(avg_y)

    # Cluster to find 12 columns
    x_coords = circles[:, 0].reshape(-1, 1)
    Z_x = linkage(x_coords, method='ward')
    col_labels = fcluster(Z_x, COLUMNS, criterion='maxclust')

    # Compute column averages
    column_averages = []
    for label in sorted(np.unique(col_labels)):
        col_circles = circles[col_labels == label]
        
        # Remove outliers from average calculation
        filtered_cols = remove_outliers(col_circles[:, 0])
        if len(filtered_cols) > 0:
            avg_x = int(np.mean(filtered_cols))
            column_averages.append(avg_x)
    
    row_averages.sort()
    column_averages.sort()

    # Return average x and y for columns and rows and shrink average radius
    return row_averages, column_averages, int(average_radius)

def convert_heic_to_jpg(image_path):
    """
    Converts a HEIC image to JPEG format.
    
    Args:
        image_path (str): Path to the HEIC image.
    
    Returns:
        str: Path to the converted JPEG image.
    """
    heic_image = Image.open(image_path)
    jpeg_path = Path(image_path).with_suffix(".jpg")
    heic_image.save(jpeg_path, "JPEG")
    return str(jpeg_path)

# Linear transformation functions for RGB values in phone-captured images without backlighting (imaging box)
def adjust_red(red): return -1.5 * red + 390

def adjust_green(green): return 0.286 * green + 125.714

def adjust_blue(blue): return 1.111 * blue + 28.888

def adjust_lighting(image):
    """
    Adjusts lighting of phone-captured images using adjustment functions.

    Args:
        image (array): Image array.
    
    Returns:
        array: Adjusted image array.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    adjusted = image_rgb.copy()

    # Adjust RGB values of each pixel following linear transformations
    adjusted[:, :, 0] = np.vectorize(adjust_red)(image_rgb[:, :, 0])
    adjusted[:, :, 1] = np.vectorize(adjust_green)(image_rgb[:, :, 1])
    adjusted[:, :, 2] = np.vectorize(adjust_blue)(image_rgb[:, :, 2])
    adjusted_bgr = cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR)

    return adjusted_bgr

def segment_image(image_path, output_folder, isUpload=True):
    """
    Segments wells from the provided image and saves the segmented wells to the output folder.
    If the input image is in HEIC format, it is first converted to JPEG.
    
    Args:
        image_path (str): Path to the input image.
        output_folder (str): Path to save the segmented wells.
    
    Returns:
        List of file paths to the segmented well images.
    """
    image_path = Path(image_path)
    if image_path.suffix.lower() == ".heic":
        print(f"Converting HEIC image: {image_path}")
        image_path = Path(convert_heic_to_jpg(image_path))

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read the image at {image_path}")
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not isUpload:
        image = adjust_lighting(image)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=140, param1=29, param2=29,
        minRadius=70, maxRadius=90
    )

    if circles is not None:
        # Format circles and group into rows in columns
        circles = np.uint16(np.around(circles))
        unique_circles = set((i[0], i[1], i[2]) for i in circles[0, :])
        rows, columns, radius = refine_grid(list(unique_circles))
        
        row_index = 1
        column_index = 1
        for y in rows:
            for x in columns:
                # Create a mask with circle
                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
                masked_circle = cv2.bitwise_and(image, mask)

                # Crop into square
                x_min = max(x - radius, 0)
                x_max = min(x + radius, image.shape[1])
                y_min = max(y - radius, 0)
                y_max = min(y + radius, image.shape[0])
                cropped_circle = masked_circle[y_min:y_max, x_min:x_max]

                # Convert background (outside the circle) to transparent
                alpha_channel = cv2.cvtColor(cropped_circle, cv2.COLOR_BGR2GRAY)
                _, alpha_channel = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
                b, g, r = cv2.split(cropped_circle)
                circle_with_alpha = cv2.merge([b, g, r, alpha_channel])

                # Save the result following index-based naming convention
                output_path = os.path.join(output_folder, f"circle_{row_index}_{column_index}.png")
                cv2.imwrite(output_path, circle_with_alpha)
                column_index += 1

            row_index += 1
            column_index = 1

    else:
        raise Exception("No circles detected")

    return rows, columns, radius
