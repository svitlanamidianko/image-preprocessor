import cv2
from rembg import remove
import os
import numpy as np
from PIL import Image

def process_image(file_path, output_folder):
    # Background removal using rembg
    with open(file_path, 'rb') as input_file:
        result = remove(input_file.read())
    temp_path = f"{output_folder}/removed_bg.png"
    with open(temp_path, 'wb') as output_file:
        output_file.write(result)

    # Load image with alpha channel
    img = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
    
    # Convert RGBA to RGB with white background
    if img.shape[2] == 4:
        # Split channels
        b, g, r, a = cv2.split(img)
        
        # Create white background
        white_bg = np.ones_like(b) * 255
        
        # Normalize alpha to range 0-1
        alpha = a.astype(float) / 255
        
        # Calculate foreground for each channel
        r = (r.astype(float) * alpha).astype(np.uint8)
        g = (g.astype(float) * alpha).astype(np.uint8)
        b = (b.astype(float) * alpha).astype(np.uint8)
        
        # Calculate background
        bg_r = (white_bg * (1.0 - alpha)).astype(np.uint8)
        bg_g = (white_bg * (1.0 - alpha)).astype(np.uint8)
        bg_b = (white_bg * (1.0 - alpha)).astype(np.uint8)
        
        # Combine foreground and background
        r = cv2.add(r, bg_r)
        g = cv2.add(g, bg_g)
        b = cv2.add(b, bg_b)
        
        # Merge channels
        img = cv2.merge([b, g, r])
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Convert to grayscale for finding bounds
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find the card
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        card_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(card_contour)
        
        # Add padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2*padding)
        h = min(height - y, h + 2*padding)
        
        # Crop the image
        cropped = img[y:y+h, x:x+w]
        
        # Calculate target dimensions while maintaining aspect ratio
        target_width = 546
        target_height = 864
        
        aspect = target_height / target_width
        current_aspect = h / w
        
        if current_aspect > aspect:
            # Image is too tall
            new_height = int(w * aspect)
            start_y = (h - new_height) // 2
            cropped = cropped[start_y:start_y+new_height, :]
        else:
            # Image is too wide
            new_width = int(h / aspect)
            start_x = (w - new_width) // 2
            cropped = cropped[:, start_x:start_x+new_width]
        
        # Final resize to target dimensions
        final = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Save the final image
        cv2.imwrite(f"{output_folder}/final_image.png", final)
    
    # Clean up temporary file
    os.remove(temp_path)

def sort_points_by_orientation(pts):
    """Sort points to maintain card orientation (top-left, top-right, bottom-right, bottom-left)"""
    # Convert points to float32
    pts = pts.astype(np.float32)
    
    # Calculate center of points
    center = np.mean(pts, axis=0)
    
    # Sort points based on their position relative to center
    top = pts[pts[:, 1] < center[1]]
    bottom = pts[pts[:, 1] >= center[1]]
    
    # Sort top and bottom points by x coordinate
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    
    # Return points in correct order
    return np.array([
        top[0],      # top-left
        top[-1],     # top-right
        bottom[-1],  # bottom-right
        bottom[0]    # bottom-left
    ])

def batch_process(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            card_output_folder = os.path.join(output_folder, filename.split('.')[0])
            os.makedirs(card_output_folder, exist_ok=True)
            
            try:
                process_image(input_path, card_output_folder)
                print(f"Processed {filename} successfully")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    input_folder = "./input_next"
    output_folder = "./output_next"
    batch_process(input_folder, output_folder)
