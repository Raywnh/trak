import pydirectinput as pydip

class SmoothCursor:
    def __init__(self, window_size=5):
        self.positions = []
        self.window_size = window_size

    def add_position(self, pos):
        if len(self.positions) >= self.window_size:
            self.positions.pop(0)
        self.positions.append(pos)

    def get_smoothed_position(self):
        if not self.positions:
            return None
        avg_x = sum(p[0] for p in self.positions) / len(self.positions)
        avg_y = sum(p[1] for p in self.positions) / len(self.positions)
        return (avg_x, avg_y)