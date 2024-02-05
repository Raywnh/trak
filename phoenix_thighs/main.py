import cv2
import image_rendering as imgr
import mouse_movement as mov

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

smoothCursor = mov.SmoothCursor(window_size=5)

while True:
    # 1.Video Capture
    ret, frame = cap.read()
    if not ret:
        break
    
    # 3. Image adjustments: gaussian blur and increasing hue and saturation for red colors only
    adjusted_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    adjusted_frame = imgr.enhance_red(adjusted_frame)
    
    # 2. Red Object Detection
    pos, contour = imgr.detect_colored_object(adjusted_frame)
    
    if pos:
        x, y = pos
        # scaled_x = x * (1920 / 640)
        # scaled_y = y * (1080 / 480)
        # smoothCursor.add_position((scaled_x, scaled_y))
        
        # smoothed_position = smoothCursor.get_smoothed_position()
        # if smoothed_position:
        #     smooth_x, smooth_y = smoothed_position
        #     pydip.moveRel(int(smooth_x), int(smooth_y))
            
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
    cv2.imshow('frame', adjusted_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows