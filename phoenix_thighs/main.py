import cv2
import image_process as imgr
import mouse_movement as mov
import pyautogui
import queue
import threading

frameQueue = queue.Queue()
resultQueue = queue.Queue()
# mouseQueue = queue.Queue()

# def mouse_thread():
#     while True:
#         if not mouseQueue.empty():
#             x, y = mouseQueue.get()
#             # pyautogui.moveTo(x, y)
        
def capture_thread(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frameQueue.put(frame)

def process_thread():
    while True:
        frame = frameQueue.get()
        frame = cv2.flip(frame, 1)
        
        if frame is None:
            break
        pos, contour = imgr.detect_colored_object(frame)
        resultQueue.put((pos, contour, frame))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 120)
smoothCursor = mov.SmoothCursor(window_size=5)

t1 = threading.Thread(target=capture_thread, args=(cap, ), daemon=True)
t2 = threading.Thread(target=process_thread, args=(), daemon=True)
# t3 = threading.Thread(target=mouse_thread, args=(), daemon=True)

t1.start()
t2.start()
# t3.start()

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if not resultQueue.empty():
        pos, contour, frame = resultQueue.get()
        
        if pos:
            x, y = pos
            
            scaled_x = x * (1920 / 640)
            scaled_y = y * (1080 / 480)
            # smoothCursor.add_position((scaled_x, scaled_y))
            # mouseQueue.put(smoothCursor.get_smoothed_position())

            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            
        cv2.imshow('frame', frame)
        
cap.release()
cv2.destroyAllWindows
t1.join()
t2.join()
# t3.join()