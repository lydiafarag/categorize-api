import cv2
import numpy as np
import joblib
import logging

# Load trained random forest model
rf_model = joblib.load("random_forest_model.joblib")

def calc_peaks(histogram):
    histogram = histogram.flatten()[1:]
    return np.argmax(histogram)

def calculate_rgb_histogram(image):
    peaks = {}
    for i, color in enumerate(['b', 'g', 'r']):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        peaks[color] = calc_peaks(histogram)
    return peaks

def rule_based_classifier(image):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        peaks = calculate_rgb_histogram(rgb_image)
        b, g, r = peaks['b'], peaks['g'], peaks['r']
        distance = np.sqrt((b - g)**2 + (g - r)**2 + (b - r)**2)
        return "Effective" if distance < 20 else "Ineffective", None
    except Exception as e:
        logging.error(f"Rule-based classification error: {e}")
        return None, e

def extract_features_for_rf(image):
    # Simple RGB histogram peak features
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    peaks = calculate_rgb_histogram(rgb_image)
    return np.array([[peaks['r'], peaks['g'], peaks['b']]])

def random_forest_classifier(image):
    try:
        features = extract_features_for_rf(image)
        prediction = rf_model.predict(features)[0]
        return prediction, None
    except Exception as e:
        logging.error(f"Random Forest classification error: {e}")
        return None, e

def rule_and_thresholding_classifier(image):
    # Run both classifiers
    rule_result, rule_error = rule_based_classifier(image)
    rf_result, rf_error = random_forest_classifier(image)

    # Decision logic
    if rule_result is None and rf_result is not None:
        return rf_result
    if rf_result is None and rule_result is not None:
        return rule_result
    if rule_result == rf_result:
        return rf_result

    # Disagreement: run second inference (optional repeat)
    # (For this example, we just rerun once)
    rule_result_2, _ = rule_based_classifier(image)
    rf_result_2, _ = random_forest_classifier(image)

    if rule_result_2 == rf_result_2:
        return rf_result_2
    else:
        # Prioritize Random Forest result
        return rf_result_2 or rf_result  # fallback in case rf_result_2 is None
