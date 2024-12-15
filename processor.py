import cv2
import numpy as np
import os

def process_card_image(image_path, debug=True):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Create debug directory if needed
    if debug:
        os.makedirs('debug', exist_ok=True)
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Broader range for background detection
    lower_bg = np.array([0, 0, 100])  # Light colors
    upper_bg = np.array([180, 30, 255])  # High value, low saturation
    
    # Create mask
    mask = cv2.inRange(hsv, lower_bg, upper_bg)
    mask = cv2.bitwise_not(mask)
    
    if debug:
        cv2.imwrite('debug/1_mask.jpg', mask)
    
    # Noise removal
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    if debug:
        cv2.imwrite('debug/2_mask_cleaned.jpg', mask)
    
    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)
    
    if debug:
        cv2.imwrite('debug/3_masked_result.jpg', result)
    
    # Find contours
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        card_contour = max(contours, key=cv2.contourArea)
        
        # Draw contour for debugging
        if debug:
            contour_image = image.copy()
            cv2.drawContours(contour_image, [card_contour], -1, (0, 255, 0), 3)
            cv2.imwrite('debug/4_contours.jpg', contour_image)
        
        # Get rotated rectangle
        rect = cv2.minAreaRect(card_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        src_pts = box.astype("float32")
        # Coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height-1],
                           [0, 0],
                           [width-1, 0],
                           [width-1, height-1]], dtype="float32")
        
        # The perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

def main():
    # Process the image
    image_path = 'input/IMG_7631.jpeg'
    processed_card = process_card_image(image_path)

    if processed_card is not None:
        # Save the result
        cv2.imwrite('output_next/processed_card.jpg', processed_card)
        print("Successfully processed and saved the card image.")
    else:
        print("Failed to process the card image.")

if __name__ == "__main__":
    main()
