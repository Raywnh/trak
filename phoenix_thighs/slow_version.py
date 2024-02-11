import cv2
import image_process as imgr
import mouse_movement as mov
import pyautogui
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 120)
smoothCursor = mov.SmoothCursor(window_size=5)


while True:
    # 1.Video Capture
    ret, frame = cap.read()
    if not ret:
        break

    adjusted_frame = cv2.flip(frame, 1)
    # Facial
    # gray = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(adjusted_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # 2. Red Object Detection
    adjusted_frame = imgr.detect_colored_object(adjusted_frame)
    
    # if pos:
    #     x, y = pos
    #     # 4. mouse movement translation
    #     scaled_x = x * (1920 / 640)
    #     scaled_y = y * (1080 / 480)
    #     smoothCursor.add_position((scaled_x, scaled_y))
        
    #     smoothed_position = smoothCursor.get_smoothed_position()
    #     if smoothed_position:
    #         smooth_x, smooth_y = smoothed_position
    #         pyautogui.moveTo(int(smooth_x), int(smooth_y))
            
    #     cv2.drawContours(adjusted_frame, [contour], -1, (0, 255, 0), 3)
    #     cv2.circle(adjusted_frame, (x, y), 5, (255, 0, 0), -1)
    # cv2.imshow('frame', frame)
    cv2.imshow('adjusted_frame', adjusted_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows