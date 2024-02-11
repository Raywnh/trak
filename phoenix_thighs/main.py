import cv2
import image_process as imgr
import mouse_movement as mov
import queue
import threading
from collections import deque
import serial
import time
import temporal_smoothing_algo as tmpa
# arduino = serial.Serial('COM7', 9600, timeout=0.1)
# time.sleep(2)

frameQueue = queue.Queue()
resultQueue = queue.Queue()
# mouseQueue = queue.Queue()

# def mouse_thread():
#     while True:
#         pos = mouseQueue.get()
#         if pos:
#             x, y = pos
#             coords = "{},{}\n".format(x, y)
#             print(coords)
#             arduino.write(coords.encode())
        
        
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

def initialize_tracker(tracker, cap):
    success, img = cap.read()
    bbox = cv2.selectROI("Tracking", img, False)
    tracker.init(img, bbox)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 120)
time.sleep(2)
# tracker = cv2.legacy.TrackerMOSSE_create()
# initialize_tracker(tracker, cap)
  
smoothCursor = mov.SmoothCursor(window_size=5)
prev_x = None
prev_y = None
wma_filter = tmpa.WeightedMovingAverage()
sma_filter = tmpa.AverageBoundingBoxTracker(window_size=10)

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
            x, y, w, h = pos
            sma_filter.update(tmpa.BoundingBox(x, y, w, h))
            smooth_bbox = sma_filter.get_smoothed_bounding_box()
            if smooth_bbox:
                center_x = smooth_bbox.x + w // 2
                center_y = smooth_bbox.y + h // 2
                
                cv2.rectangle(frame, (smooth_bbox.x, smooth_bbox.y), 
                    (smooth_bbox.x + smooth_bbox.w, smooth_bbox.y + smooth_bbox.h), 
                    (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
        
            scaled_x = (x / 640) * 1920
            scaled_y = (y / 480) * 1080
            
            # smoothCursor.add_position((scaled_x, scaled_y))
            # smooth_position = smoothCursor.get_smoothed_position()
            # smooth_x, smooth_y = smooth_position
            
            # if prev_x is not None and prev_y is not None:
            #    move_x = scaled_x - prev_x
            #    move_y = scaled_y - prev_y
            #    mouseQueue.put((move_x, move_y))
                
            # prev_x = scaled_x
            # prev_y = scaled_y 
            
            # avg_box = bounding_box_tracker.get_average_bounding_box()
            # if avg_box:
                # cv2.rectangle(frame, (avg_box.x, avg_box.y), (avg_box.x + avg_box.w, avg_box.y + avg_box.h), (0, 255, 0), 2)
                # center_x = avg_box.x + avg_box.w // 2
                # center_y = avg_box.y + avg_box.w // 2
                # cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            
        cv2.imshow('frame', frame)
        
cap.release()
cv2.destroyAllWindows()
t1.join()
t2.join()
# t3.join()