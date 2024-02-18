import cv2
import image_process as imgr
import mouse_movement as mov
import queue
import threading
import serial
import time
import temporal_smoothing_algo as tmpa
arduino = serial.Serial('COM7', 9600)

frameQueue = queue.Queue()
resultQueue = queue.Queue()
mouseQueue = queue.Queue()

def mouse_thread():
    while True:
        pos = mouseQueue.get()
        if pos:
            x, y = pos
            
            coords = "{},{}\n".format(x, y)
            arduino.write(coords.encode())
            print(coords)
        
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
        pos, centroid = imgr.detect_colored_object(frame)
        resultQueue.put((pos, centroid, frame))

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 60)
time.sleep(2)
  
smoothCursor = mov.SmoothCursor(window_size=5)
prev_x = None
prev_y = None

sma_filter = tmpa.AverageBoundingBoxTracker(window_size=10, min_movement_threshold=10)

t1 = threading.Thread(target=capture_thread, args=(cap, ), daemon=True)
t2 = threading.Thread(target=process_thread, args=(), daemon=True)
t3 = threading.Thread(target=mouse_thread, args=(), daemon=True)

t1.start()
t2.start()
t3.start()

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if not resultQueue.empty():
        pos, centroid, frame = resultQueue.get()
        
        if pos and centroid:
            x, y, w, h = pos
            centroid_x, centroid_y = centroid
            sma_filter.update(tmpa.BoundingBox(centroid_x, centroid_y, w, h))
            smooth_bbox = sma_filter.get_smoothed_bounding_box()
            
            if smooth_bbox:
                
                # cv2.rectangle(frame, (centroid_x, centroid_y), 
                #     (centroid_x + smooth_bbox.w, centroid_y + smooth_bbox.h), 
                #     (0, 255, 0), 2)
                cv2.circle(frame, (smooth_bbox.x, smooth_bbox.y), 50, (0, 255, 0), 1)

                scaled_x = int((smooth_bbox.x / 640) * 1920)
                scaled_y = int((smooth_bbox.y / 480) * 1080)
                
                if not prev_x and not prev_y:
                    prev_x = scaled_x
                    prev_y = scaled_y
                
                move_x = scaled_x - prev_x
                move_y = scaled_y - prev_y
                
                prev_x = scaled_x
                prev_y = scaled_y
                
                mouseQueue.put((move_x, move_y))
                
        cv2.imshow('frame', frame)
        
cap.release()
cv2.destroyAllWindows()

t1.join()
t2.join()
t3.join()