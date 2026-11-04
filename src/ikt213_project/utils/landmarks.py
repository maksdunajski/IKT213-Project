import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark
import cv2  # må importeres etter mediapipe, ellers blir prosessen drept (exit 137)

def frameToMpImage(frame):
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)

def landmarks_to_features(hand, handedness_label, width, height):
    pts = np.array([[lm.x * width, lm.y * height, lm.z * width] for lm in hand])
    pts -= pts[0]                                  # relativt til håndledd
    if handedness_label == 'Left':
        pts[:, 0] *= -1                            # speil til felles side
    pts /= np.linalg.norm(pts[9]) + 1e-6           # skaler med håndstørrelse
    return pts.flatten()                           # vektor med 63 verdier


def extractHands(frame, detector):
    height, width, _ = frame.shape # mp returnerer verdier mellom 0 og 1 og ikke koordinater
    result = detector.detect(frameToMpImage(frame))

    hands = []
    for hand, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name
        hands.append({
            "hand": label,
            "score": handedness[0].score,
            "features": landmarks_to_features(hand, label, width, height),
            "landmarks": hand,
        })
    return hands

def drawBoundingBox(landmarks, width, height, pad=20):
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]

    x_min = max(int(min(xs)) - pad, 0)
    y_min = max(int(min(ys)) - pad, 0)
    x_max = min(int(max(xs)) + pad, width)
    y_max = min(int(max(ys)) + pad, height)

    return x_min, y_min, x_max, y_max