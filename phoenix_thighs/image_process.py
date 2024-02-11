import cv2
import numpy as np

def detect_colored_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.medianBlur(frame, 5)
    
    lower_red1 = np.array([0, 160, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 160, 70])
    upper_red2 = np.array([180, 255, 255])
    
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 3. Make red clearer with morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
    
        return (x, y, w, h), largest_contour
    return None, None