# Setup

```bash
./setup/model.sh
./setup/data.sh
```

# Landmark extraction
```bash
uv run src/ikt213_project/utils/extract_landmarks.py
```

# Frontend
```bash
uv run ikt213-frontend
```
Then open http://127.0.0.1:5001. Use `--camera` to pick another camera and `--port` to change the port.

The model menu lists every `.pkl` file in `src/ikt213_project/model/trained_models/`. Reload the page to see new files.
