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