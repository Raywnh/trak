import numpy as np
import cv2
from collections import deque

class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
class AverageBoundingBoxTracker:
    def __init__(self, window_size=5, min_movement_threshold=10.0):
        self.window_size = window_size
        self.bounding_boxes = deque(maxlen=window_size)
        self.movement_threshold = min_movement_threshold
        self.prev_center = None
        
    def update(self, new_box):
        self.bounding_boxes.append(new_box)
        
    def get_smoothed_bounding_box(self):
        if not self.bounding_boxes:
            return None
        
        avg_x = int(np.mean([box.x for box in self.bounding_boxes]))
        avg_y = int(np.mean([box.y for box in self.bounding_boxes]))
        avg_w = int(np.mean([box.w for box in self.bounding_boxes]))
        avg_h = int(np.mean([box.h for box in self.bounding_boxes]))
        
        if self.prev_center is not None:
            dx = abs(self.prev_center[0] - avg_x)
            dy = abs(self.prev_center[1] - avg_y)
            distance_moved = (dx ** 2 + dy ** 2) ** 0.5
            
            if distance_moved < self.movement_threshold:
                return BoundingBox(self.prev_center[0], self.prev_center[1], self.prev_center[2], self.prev_center[3])
        
        self.prev_center = (avg_x, avg_y, avg_w, avg_h)
        return BoundingBox(avg_x, avg_y, avg_w, avg_h)
    
class ExponentialMovingAverage:
    def __init__(self, alpha=0.1, min_threshold=1.0):
        self.alpha = alpha
        self.smoothed_bbox = None
        self.threshold = min_threshold
        self.prev_center = None

    def update(self, bbox):
        if self.smoothed_bbox is None:
            self.smoothed_bbox = bbox
            return bbox
        else:
            x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            smoothed_x = int(self.alpha * x + (1 - self.alpha) * self.smoothed_bbox.x)
            smoothed_y = int(self.alpha * y + (1 - self.alpha) * self.smoothed_bbox.y)
            smoothed_w = int(self.alpha * w + (1 - self.alpha) * self.smoothed_bbox.w)
            smoothed_h = int(self.alpha * h + (1 - self.alpha) * self.smoothed_bbox.h)
            
            if self.prev_center is not None:
                dx = abs(self.prev_center[0] - smoothed_x)
                dy = abs(self.prev_center[1] - smoothed_y)
                distance_moved = (dx ** 2 + dy ** 2) ** 0.5
                
                if distance_moved < self.threshold:
                    return BoundingBox(self.prev_center[0], self.prev_center[1], self.prev_center[2], self.prev_center[3])
        
            self.prev_center = (smoothed_x, smoothed_y, smoothed_w, smoothed_h)
            return BoundingBox(smoothed_x, smoothed_y, smoothed_w, smoothed_h)
    
class KalmanFilter:
    def __init__(self, process_noise_cov=1e-2, measurement_noise_cov=1e-1, error_cov_post=0.1):
        # State - what it tracks (4 elements -> x, y, x velocity, y velocity but measure only 2 elements - x, y)
        self.kalman_filter = cv2.KalmanFilter(4, 2) 
        # How to map measurements to state model (only measuring x and y positions, not velocities)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # Models how object moves from one state to the next (this matrix says next state depends on current position and velocity)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # How much noise to expect in object motion, higher means expect more random movement
        self.kalman_filter.processNoiseCov = process_noise_cov * np.eye(4, dtype=np.float32)
        # How much noise to expect in measruements, more means less confident in measurements
        self.kalman_filter.measurementNoiseCov = measurement_noise_cov * np.eye(2, dtype=np.float32)
        # Initial guess of how accurate state estimate is, higher number means less sure about initial position and velocity estimates
        self.kalman_filter.errorCovPost = error_cov_post * np.eye(8, dtype=np.float32)
    
  
    def predict(self, x, y):
        positions = np.array([[np.float32(x)], [np.float32(y)]])
        # Updating estimates of object position and  velocity
        self.kalman_filter.correct(positions)
        # Predicts using measurements and estimates
        prediction = self.kalman_filter.predict()
        return int(prediction[0]), int(prediction[1])