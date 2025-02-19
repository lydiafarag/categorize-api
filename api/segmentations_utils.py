import numpy as np
import cv2
import os
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF opener for PIL to handle HEIC files
register_heif_opener()

BUFFER = 30  # Buffer space for row/column sorting

def group_circles(circles, buffer=BUFFER):
    sorted_circles = sorted(circles, key=lambda c: (c[1], c[0]))

    # Group circles by rows
    rows = []
    current_row = []
    previous_y = sorted_circles[0][1]  # Start with the first circle's y position
    
    for x, y, r in sorted_circles:
        if abs(y - previous_y) <= buffer:  # If y is within the buffer, it's the same row
            current_row.append((x, y, r))
        else:
            # If not, push the current row to rows and start a new row
            rows.append(current_row)
            current_row = [(x, y, r)]
        previous_y = y
    
    # Add the last row to the rows list
    if current_row:
        rows.append(current_row)

def group_circles_by_zone(circles, buffer=BUFFER):
    """
    Groups circles into rows and columns based on the proximity (buffer) of their y and x values, respectively.
    Fills missing circles with the average x and y values for their respective rows and columns.
    """
    
    # Group circles by rows using the y-position (within the BUFFER)
    sorted_circles = sorted(circles, key=lambda c: (c[1], c[0]))  # Sort by y first, then x
    rows = []
    current_row = []
    previous_y = sorted_circles[0][1]
    
    for x, y, r in sorted_circles:
        if abs(y - previous_y) <= buffer:
            current_row.append((x, y, r))
        else:
            rows.append(current_row)
            current_row = [(x, y, r)]
        previous_y = y
    
    if current_row:
        rows.append(current_row)

    # Group circles by columns using the x-position (within the BUFFER)
    sorted_circles = sorted(circles, key=lambda c: (c[0], c[1]))  # Sort by x first, then y
    columns = []
    current_column = []
    previous_x = sorted_circles[0][0]
    
    for x, y, r in sorted_circles:
        if abs(x - previous_x) <= buffer:
            current_column.append((x, y, r))
        else:
            columns.append(current_column)
            current_column = [(x, y, r)]
        previous_x = x
    
    if current_column:
        columns.append(current_column)

    # Calculate expected positions for rows and columns using averages
    row_averages = []
    for row in rows:
        avg_y = np.mean([circle[1] for circle in row])
        row_averages.append(int(avg_y))

    column_averages = []
    for col in columns:
        avg_x = np.mean([circle[0] for circle in col])
        column_averages.append(int(avg_x))

    # Compute average radius
    radii = [circle[2] for circle in circles]
    average_radius = 0.8 * sum(radii) / len(radii) if radii else 0

    # Compute final circle positions using intersections of average rows and columns and average radius
    final_grid = []
    for row in row_averages:
        average_row = []
        for col in column_averages:
            average_row.append((int(col), int(row), int(average_radius)))
        final_grid.append(average_row)

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

def segment_image(image_path, output_folder):
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
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=140, param1=29, param2=29,
        minRadius=70, maxRadius=90
    )

    if circles is not None:
        # Convert circles to integer values
        circles = np.uint16(np.around(circles))

        # Get unique circles (no duplicates)
        unique_circles = set((i[0], i[1], i[2]) for i in circles[0, :])

        # Group the circles into rows and columns based on the 20-pixel buffer
        rows, columns, radius = group_circles_by_zone(list(unique_circles), buffer=BUFFER)
        
        row_index = 1
        column_index = 1

        for y in rows:
            for x in columns:

                # Create a mask with circle
                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
                masked_circle = cv2.bitwise_and(image, mask)

                # Calculate square bounds and crop
                x_min = max(x - radius, 0)
                x_max = min(x + radius, image.shape[1])
                y_min = max(y - radius, 0)
                y_max = min(y + radius, image.shape[0])
                cropped_circle = masked_circle[y_min:y_max, x_min:x_max]

                # Convert black background (outside the circle) to transparent
                alpha_channel = cv2.cvtColor(cropped_circle, cv2.COLOR_BGR2GRAY)
                _, alpha_channel = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
                b, g, r = cv2.split(cropped_circle)
                circle_with_alpha = cv2.merge([b, g, r, alpha_channel])

                # Save the result with index-based naming
                output_path = os.path.join(output_folder, f"circle_{row_index}_{column_index}.png")
                cv2.imwrite(output_path, circle_with_alpha)
                column_index += 1

            row_index += 1
            column_index = 1

    else:
        raise Exception("No circles detected")

    return rows, columns, radius