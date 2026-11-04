"""Find the trained models on disk and switch between them while the app runs."""
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).resolve().parents[1] / "model" / "trained_models"
FEATURE_COUNT = 63  # 21 landmarks * (x, y, z)


class ModelLoadError(Exception):
    pass


@dataclass(frozen=True)
class Prediction:
    letter: str
    confidence: float


class SklearnPredictor:
    """Any pickled scikit-learn classifier, such as knn_model.pkl."""

    def __init__(self, model):
        self.model = model
        # knn.py fits on a DataFrame, so the model expects the same column names
        self.columns = getattr(model, "feature_names_in_", None)

    def predict(self, features):
        X = features.reshape(1, -1)
        if self.columns is not None:
            X = pd.DataFrame(X, columns=self.columns)
        probabilities = self.model.predict_proba(X)[0]
        best = int(probabilities.argmax())
        return Prediction(str(self.model.classes_[best]), float(probabilities[best]))


class CNN(nn.Module):
    """Must match the CNN class in model/cnn-landmarks.py.

    It is repeated here because importing that script would start training.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CnnLandmarksPredictor:
    """A pickled {"state_dict": ..., "classes": [...]} saved by cnn-landmarks.py."""

    def __init__(self, checkpoint):
        self.classes = [str(c) for c in checkpoint["classes"]]
        self.model = CNN(len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, features):
        # (63,) -> (1, 3, 21): x/y/z are the channels, as in LandmarksDataset
        x = torch.as_tensor(features, dtype=torch.float32).reshape(1, 21, 3).permute(0, 2, 1)
        with torch.no_grad():
            probabilities = F.softmax(self.model(x), dim=1)[0]
        best = int(probabilities.argmax())
        return Prediction(self.classes[best], float(probabilities[best]))


def load_model(path):
    """Load a .pkl file and check that it can make a prediction."""
    try:
        with open(path, "rb") as file:
            saved = pickle.load(file)

        if isinstance(saved, dict) and "state_dict" in saved:
            predictor = CnnLandmarksPredictor(saved)
        elif hasattr(saved, "predict_proba") and hasattr(saved, "classes_"):
            predictor = SklearnPredictor(saved)
        else:
            raise TypeError(
                "expected a fitted scikit-learn classifier or a dict with "
                f"'state_dict' and 'classes', got {type(saved).__name__}"
            )

        # A model trained on other features fails here, not in the middle of the video feed
        predictor.predict(np.zeros(FEATURE_COUNT, dtype=np.float32))
    except Exception as error:  # unpickling can raise almost anything
        raise ModelLoadError(f"Could not load {path.name}: {error}") from error
    return predictor


class ModelManager:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._lock = threading.Lock()
        self._active_name = None
        self._predictor = None

    def available(self):
        """Model file names, read from disk each time so new .pkl files are picked up."""
        return sorted(path.name for path in self.models_dir.glob("*.pkl"))

    @property
    def active(self):
        with self._lock:
            return self._active_name

    def switch_model(self, name):
        """Make `name` the active model. If it fails, the current model stays active."""
        # Checking against the listing also stops a request from reaching other paths
        if name not in self.available():
            raise ModelLoadError(f"Unknown model: {name}")
        # Load outside the lock so the video feed keeps running on the old model
        predictor = load_model(self.models_dir / name)
        with self._lock:
            self._active_name = name
            self._predictor = predictor

    def load_first_available(self):
        for name in self.available():
            try:
                self.switch_model(name)
                return
            except ModelLoadError as error:
                print(error)

    def predict(self, features):
        with self._lock:
            predictor = self._predictor
        return predictor.predict(features) if predictor else None
