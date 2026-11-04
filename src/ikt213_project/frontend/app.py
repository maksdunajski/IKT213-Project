"""Flask frontend: live camera feed with landmarks, predicted letter and model switching."""
import argparse

# video must be imported first: it loads mediapipe before cv2 (see the note in video.py)
from ikt213_project.frontend.video import LANDMARKER_PATH, VideoProcessor, cv2, create_detector
from ikt213_project.frontend.models import ModelLoadError, ModelManager
from flask import Flask, Response, jsonify, render_template, request


def create_app(processor, models):
    app = Flask(__name__)

    def state_json():
        return {**processor.state, "model": models.active}

    @app.get("/")
    def index():
        return render_template("index.html", models=models.available(), active=models.active)

    @app.get("/video_feed")
    def video_feed():
        return Response(
            processor.mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/state")
    def state():
        return jsonify(state_json())

    @app.post("/api/model")
    def select_model():
        name = (request.get_json(silent=True) or {}).get("name")
        try:
            models.switch_model(name)
        except ModelLoadError as error:
            return jsonify({"error": str(error), "model": models.active}), 400
        return jsonify(state_json())

    return app


def main():
    parser = argparse.ArgumentParser(description="Run the sign language frontend")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    # Not 5000: AirPlay Receiver usually occupies that port on macOS
    parser.add_argument("--port", type=int, default=5001, help="port (default: 5001)")
    args = parser.parse_args()

    if not LANDMARKER_PATH.exists():
        raise SystemExit(f"Missing {LANDMARKER_PATH}. Run ./setup/model.sh first.")

    # Open the camera on the main thread: macOS can only ask for camera permission from here
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. On macOS, check that your terminal "
            "has access under System Settings > Privacy & Security > Camera."
        )

    models = ModelManager()
    models.load_first_available()
    if models.active is None:
        print(f"No usable .pkl model in {models.models_dir}, so no letters will be predicted.")

    processor = VideoProcessor(capture, create_detector(), models)
    processor.start()

    try:
        # The reloader would start a second process that also tries to open the camera
        create_app(processor, models).run(
            host="127.0.0.1", port=args.port, threaded=True, use_reloader=False
        )
    finally:
        capture.release()


if __name__ == "__main__":
    main()
