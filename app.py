import json
import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "animal_mobilenetv2.keras")
LABELS_PATH = os.path.join(APP_DIR, "models", "class_names.json")
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = None
class_names = []
if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    return [
        {"label": class_names[i].capitalize(), "confidence": float(probabilities[i]) * 100}
        for i in top_indices
    ]


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None
    uploaded_image = None

    if request.method == "POST":
        if model is None:
            error = "Model files are missing. Run train.py first."
        elif "animal_image" not in request.files:
            error = "Please choose an image."
        else:
            file = request.files["animal_image"]
            if not file or file.filename == "":
                error = "Please choose an image."
            elif not allowed_file(file.filename):
                error = "Unsupported file type. Upload a PNG, JPG, or JPEG image."
            else:
                filename = secure_filename(file.filename)
                # Avoid overwriting a previous upload with the same name.
                safe_name = f"{os.urandom(8).hex()}_{filename}"
                path = os.path.join(UPLOAD_DIR, safe_name)
                try:
                    file.save(path)
                    results = predict_image(path)
                    uploaded_image = safe_name
                except (UnidentifiedImageError, OSError, ValueError):
                    error = "The uploaded file could not be read as an image."
                finally:
                    # Prediction is complete; remove the uploaded file from disk.
                    if os.path.exists(path):
                        os.remove(path)

    return render_template(
        "index.html", results=results, error=error, uploaded_image=uploaded_image,
        model_ready=model is not None
    )


if __name__ == "__main__":
    app.run(debug=True)
