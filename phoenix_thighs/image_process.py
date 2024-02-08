import cv2
import numpy as np

def detect_colored_object(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 140, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 140, 30])
    upper_red2 = np.array([180, 255, 255])
    
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 3. Make red clearer with morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    if contours:
        largest_contour = max(contours, key = cv2.contourArea)
        (cX, cY), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(cX), int(cY))
        radius = int(radius)
        
        # Draw the circle on the image
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (255, 255, 255), -1)

        # Return the center coordinates and radius of the circle
        return center, largest_contour
    return None, None