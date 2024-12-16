import cv2
import numpy as np
import os

def find_card_contour(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (which should be the card)
    largest_contour = max(contours, key=cv2.contourArea)
    
    return largest_contour

def crop_card(image_path, output_path):
    # Read image
    image = cv2.imread(image_path)
    
    # Find the card contour
    card_contour = find_card_contour(image)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(card_contour)
    
    # Crop the image
    cropped = image[y:y+h, x:x+w]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped)

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process all images in input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f'cropped_{filename}')
            
            try:
                crop_card(input_path, output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Usage
input_folder = "input_images"
output_folder = "output_images"
process_folder(input_folder, output_folder)