from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sqlite3
import gdown

app = Flask(__name__)

# Load trained model



MODEL_PATH = "model/pneumonia_model.keras"

if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    gdown.download(
        "https://drive.google.com/uc?id=1X9jFJfrV8SlEgCwnT9u1hdijLPxUWXLT",
        MODEL_PATH,
        quiet=False
    )
    model = tf.keras.models.load_model(MODEL_PATH)


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        result = "PNEUMONIA"
        confidence = confidence * 100
    else:
        result = "NORMAL"
        confidence = (1 - confidence) * 100

    return result, round(confidence, 2)
def save_prediction(image_name, prediction):
    
    conn = sqlite3.connect("database/medical.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO predictions(image_name, prediction) VALUES (?, ?)",
        (image_name, prediction)
    )

    conn.commit()
    conn.close()    
def get_predictions():
    conn = sqlite3.connect("database/medical.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")

    rows = cursor.fetchall()

    conn.close()

    return rows    


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    image_name = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image_name = file.filename

            result, confidence = predict_image(filepath)
            save_prediction(file.filename, result)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,

        image_name=image_name
    )

from flask import jsonify

@app.route("/history")
def history():

    predictions = get_predictions()

    return render_template(
        "history.html",
        predictions=predictions
    )

@app.route("/api/health", methods=["GET"])
@app.route("/api/predict", methods=["POST"])
def api_predict():

    file = request.files.get("image")

    if not file:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = predict_image(filepath)

    save_prediction(file.filename, result)

    return jsonify({
        "image": file.filename,
        "prediction": result
    })
def health():
    return jsonify({
        "status": "success",
        "message": "Advanced AI Medical Intelligence Platform API is running"
    })

if __name__ == "__main__":
    app.run(debug=True)