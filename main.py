import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark
import cv2  # må importeres etter mediapipe, ellers blir prosessen drept (exit 137)
from ikt213_project.utils.landmarks import extractHands, drawBoundingBox


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Kamera åpnet:", cap.isOpened())
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fikk ikke lest frame fra kameraet, avslutter")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape # mp returnerer verdier mellom 0 og 1 og ikke koordinater

        for h in extractHands(frame, detector):
            x_min, y_min, x_max, y_max = drawBoundingBox(h["landmarks"], width, height)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0),2)
            print(h["hand"], h["features"].shape)
            for lm in h["landmarks"]:
                px = int(lm.x * width)
                py = int(lm.y * height)
                cv2.circle(frame, (px,py), 5, (0, 255, 255), 2)

        cv2.imshow("Webcam feed", frame)
        # print(detectionResult)
        # print(frame.shape)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()