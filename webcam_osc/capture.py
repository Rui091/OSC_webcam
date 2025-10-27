import cv2
from typing import Optional
import numpy as np


class WebcamCapture:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()

    def get_frame(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> 'WebcamCapture':
        self.start()
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: type) -> None:
        self.stop()
