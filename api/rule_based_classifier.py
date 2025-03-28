import cv2
import numpy as np

def calc_peaks(histogram):
    histogram = histogram.flatten()[1:]
    return np.argmax(histogram)

def calculate_rgb_histogram(image):
    peaks = {}
    for i, color in enumerate(['b', 'g', 'r']):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        peaks[color] = calc_peaks(histogram)
    return peaks

def classify_by_rgb_peaks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    peaks = calculate_rgb_histogram(rgb_image)
    b, g, r = peaks['b'], peaks['g'], peaks['r']
    distance = np.sqrt((b - g)**2 + (g - r)**2 + (b - r)**2)

    if distance < 10:
        return "Effective"
    elif 10 <= distance < 25:
        return "Somewhat Effective"
    else:
        return "Ineffective"
