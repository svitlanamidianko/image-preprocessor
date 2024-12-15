import cv2
import numpy as np

def process_card(image_path, output_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Could not read the image")

    # Step 1: Remove background
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for red background
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    mask = cv2.bitwise_not(mask)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # Step 2: Find card contour and correct perspective
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise Exception("No card contour found")
    
    card_contour = max(contours, key=cv2.contourArea)
    
    rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    width = int(rect[1][0])
    height = int(rect[1][1])
    
    if width > height:
        width, height = height, width
    
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    src_points = np.float32(box)
    src_points = order_points(src_points)
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    final = cv2.warpPerspective(result, M, (width, height))
    
    cv2.imwrite(output_path, final)
    return final

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect 