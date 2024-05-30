from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easygui
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
import re

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def get_detected_objects(image, model, threshold=0.4):
    results = model(image)[0]
    detected_objects = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            detected_objects.append({
                'class': results.names[int(class_id)],
                'confidence': score,
                'relative_coordinates': {
                    'x1': x1 / image.shape[1],
                    'y1': y1 / image.shape[0],
                    'x2': x2 / image.shape[1],
                    'y2': y2 / image.shape[0]
                }
            })

    return detected_objects

def normalize_bbox(bbox, image_width, image_height):
    return [coord / image_width if i % 2 == 0 else coord / image_height for i, coord in enumerate(bbox)]

def bbox_cosine_similarity(bbox1, bbox2):
    return cosine_similarity([bbox1], [bbox2])[0][0]

def are_objects_similar(user_obj, template_obj, image_width, image_height):
    class_similarity_threshold = 0
    position_similarity_threshold = 0.9

    class_similarity = 1 if user_obj['class'] == template_obj['class'] else 0

    user_normalized_bbox = normalize_bbox(user_obj['relative_coordinates'].values(), image_width, image_height)
    template_normalized_bbox = normalize_bbox(template_obj['relative_coordinates'].values(), image_width, image_height)
    position_similarity = bbox_cosine_similarity(user_normalized_bbox, template_normalized_bbox)

    return class_similarity > class_similarity_threshold and position_similarity > position_similarity_threshold

def extract_and_display_text(image):
    data = pytesseract.image_to_data(image, config='--psm 6', output_type=pytesseract.Output.DICT)

    for i in range(len(data['text'])):
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        confidence = int(data['conf'][i])
        text = data['text'][i]

        if confidence > 60 and w > 5 and h > 5 and w / h > 0.2:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"Text: {text}, Confidence: {confidence}, Bounding Box: ({x}, {y}, {x + w}, {y + h})")

    return image, data

def is_pan_number(text):
    pan_pattern = r'^[A-Z]{5}\d{4}[A-Z]$'
    return re.match(pan_pattern, text) is not None

def display_result_with_boxes(image, data):
    for i in range(len(data['text'])):
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        confidence = int(data['conf'][i])
        text = data['text'][i]

        if confidence > 60 and w > 5 and h > 5 and w / h > 0.2:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{text} ({confidence}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Result with Text and Bounding Boxes')
    plt.show()

# Load images and model
user_input_image = cv2.imread('PAN/bgremoved.jpg')
template_image = cv2.imread('PAN/pan.jpg')
model = YOLO('PAN/pan.pt')

# Get detected objects for user-input image and template image
print("Detections from user input")
user_objects = get_detected_objects(user_input_image, model)
print("\nDetections from template image")
template_objects = get_detected_objects(template_image, model)

# Compare detected objects
similar_objects_count = sum(1 for user_obj in user_objects for template_obj in template_objects
                           if are_objects_similar(user_obj, template_obj, user_input_image.shape[1], user_input_image.shape[0]))

# Save the output image with bounding boxes for user input
for user_obj in user_objects:
    x1, y1, x2, y2 = [int(coord * user_input_image.shape[i % 2]) for i, coord in enumerate(user_obj['relative_coordinates'].values())]
    cv2.rectangle(user_input_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

# Display the final user-input image if more than half of the classes are similar
if similar_objects_count > len(user_objects) / 2:
    user_input_image, text_data = extract_and_display_text(user_input_image)

    # Check if PAN number is detected
    for text in text_data['text']:
        if is_pan_number(text):
            print(f"PAN Number Detected: {text}\n")

    # Display the result with text and bounding boxes
    display_result_with_boxes(user_input_image, text_data)
    print("Pan Card Detected")
else:
    print("Not a Pan Card")
