import base64
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = os.path.join("static", "outputs")
MODEL_PATH = "yolov8n.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)


def make_counts(result):
    """Convert YOLO class ids into a simple object-count dictionary."""
    counts = Counter()

    if result.boxes is None:
        return {}

    for class_id in result.boxes.cls:
        object_name = model.names[int(class_id)]
        counts[object_name] += 1

    return dict(sorted(counts.items()))


def make_alerts(counts):
    alerts = []

    if counts.get("person", 0) >= 3:
        alerts.append({
            "title": "Crowd Alert",
            "message": "Three or more people detected in the scene.",
            "type": "danger"
        })

    if counts.get("car", 0) >= 5:
        alerts.append({
            "title": "Traffic Alert",
            "message": "Five or more cars detected in the scene.",
            "type": "warning"
        })

    return alerts


def total_objects(counts):
    return sum(counts.values())


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/image", methods=["GET", "POST"])
def image_detection():
    detected_image = None
    counts = {}
    alerts = []
    message = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or uploaded_file.filename == "":
            message = "Please choose an image before running detection."
        else:
            filename = secure_filename(uploaded_file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(input_path)

            results = model(input_path)
            result = results[0]
            counts = make_counts(result)
            alerts = make_alerts(counts)

            annotated_image = result.plot()
            output_filename = f"detected_{int(time.time())}_{filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(output_path, annotated_image)

            detected_image = "/" + output_path.replace("\\", "/")
            message = "Image detection completed successfully."

    return render_template(
        "image.html",
        detected_image=detected_image,
        counts=counts,
        total=total_objects(counts),
        alerts=alerts,
        message=message
    )


@app.route("/video", methods=["GET", "POST"])
def video_detection():
    detected_video = None
    message = None

    if request.method == "POST":
        uploaded_file = request.files.get("video")

        if not uploaded_file or uploaded_file.filename == "":
            message = "Please choose a video before running detection."
        else:
            filename = secure_filename(uploaded_file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(input_path)

            try:
                mp4_path = run_yolo_and_convert_video(input_path)
                detected_video = "/" + mp4_path.replace("\\", "/")
                message = "Video detection completed successfully."
            except Exception as error:
                message = f"Video processing failed: {error}"

    return render_template(
        "video.html",
        detected_video=detected_video,
        message=message
    )


def run_yolo_and_convert_video(input_path):
    """
    Let YOLO save its detected video first, then convert that output to MP4
    so the browser can play it with the HTML5 video tag.
    """
    results = model.predict(source=input_path, save=True)
    yolo_output_path = find_yolo_video_output(results, input_path)

    input_name = Path(input_path).stem
    output_filename = f"detected_{int(time.time())}_{input_name}.mp4"
    mp4_output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    convert_video_to_mp4(yolo_output_path, mp4_output_path)

    return mp4_output_path


def find_yolo_video_output(results, input_path):
    """Find the video file saved by YOLO in the runs folder."""
    input_stem = Path(input_path).stem
    save_dir = Path(results[0].save_dir)
    video_extensions = [".avi", ".mp4", ".mov", ".mkv"]

    for extension in video_extensions:
        matching_file = save_dir / f"{input_stem}{extension}"
        if matching_file.exists():
            return str(matching_file)

    for file_path in save_dir.iterdir():
        if file_path.suffix.lower() in video_extensions:
            return str(file_path)

    raise FileNotFoundError("YOLO output video was not found.")


def convert_video_to_mp4(input_video_path, output_video_path):
    """Convert YOLO .avi output to .mp4 using OpenCV."""
    video = cv2.VideoCapture(input_video_path)

    if not video.isOpened():
        raise ValueError("Could not open YOLO output video for conversion.")

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 24

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        success, frame = video.read()
        if not success:
            break

        writer.write(frame)

    video.release()
    writer.release()

    if not os.path.exists(output_video_path):
        raise FileNotFoundError("MP4 video was not created.")


@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


@app.route("/webcam-detect", methods=["POST"])
def webcam_detect():
    data = request.get_json()
    image_data = data.get("image", "")

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    results = model(frame, verbose=False)
    result = results[0]
    counts = make_counts(result)
    alerts = make_alerts(counts)

    annotated_frame = result.plot()
    success, buffer = cv2.imencode(".jpg", annotated_frame)

    if not success:
        return jsonify({"error": "Could not process webcam frame."}), 500

    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "image": f"data:image/jpeg;base64,{encoded_image}",
        "counts": counts,
        "total": total_objects(counts),
        "alerts": alerts
    })


if __name__ == "__main__":
    app.run(debug=True)
