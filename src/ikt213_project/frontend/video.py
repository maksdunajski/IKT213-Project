"""Read the camera, draw MediaPipe landmarks and predict a letter for each frame."""
import threading
import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2  # må importeres etter mediapipe, ellers blir prosessen drept (exit 137)

from ikt213_project.utils.landmarks import drawBoundingBox, extractHands

LANDMARKER_PATH = Path(__file__).resolve().parents[3] / "hand_landmarker.task"
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


def create_detector():
    base_options = python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
    # One hand, because the page shows one predicted letter
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    return vision.HandLandmarker.create_from_options(options)


def draw_hand(frame, landmarks):
    height, width, _ = frame.shape
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    x_min, y_min, x_max, y_max = drawBoundingBox(landmarks, width, height)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 255), 2)


class VideoProcessor(threading.Thread):
    """Runs in the background so every browser tab shares one camera and one detector."""

    def __init__(self, capture, detector, models):
        super().__init__(daemon=True)
        self.capture = capture
        self.detector = detector
        self.models = models
        self._new_frame = threading.Condition()
        self._jpeg = None
        self._frame_id = 0
        self._state = {"letter": None, "confidence": None, "hand_detected": False, "error": None}

    def run(self):
        while True:
            ok, frame = self.capture.read()
            if not ok:
                self._set_state(error="Could not read from the camera")
                time.sleep(0.2)
                continue

            frame = cv2.flip(frame, 1)  # mirrored, like main.py
            hands = extractHands(frame, self.detector)
            prediction = None
            if hands:
                draw_hand(frame, hands[0]["landmarks"])
                prediction = self.models.predict(hands[0]["features"])

            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            with self._new_frame:
                self._jpeg = jpeg.tobytes()
                self._frame_id += 1
                self._state = {
                    "letter": prediction.letter if prediction else None,
                    "confidence": prediction.confidence if prediction else None,
                    "hand_detected": bool(hands),
                    "error": None,
                }
                self._new_frame.notify_all()

    def _set_state(self, **changes):
        with self._new_frame:
            self._state = {**self._state, **changes}

    @property
    def state(self):
        with self._new_frame:
            return self._state

    def mjpeg_stream(self):
        """Yield the newest frames in the multipart format browsers show in an <img>."""
        last_id = 0
        while True:
            with self._new_frame:
                self._new_frame.wait_for(lambda: self._frame_id != last_id, timeout=1)
                if self._frame_id == last_id:
                    continue
                last_id = self._frame_id
                jpeg = self._jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
