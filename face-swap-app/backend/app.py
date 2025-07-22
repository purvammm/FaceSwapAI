from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["PROCESSED_FOLDER"] = "processed"
app.config["FACESWAP_PATH"] = "../faceswap/faceswap.py"

@app.route("/api/swap", methods=["POST"])
def swap():
    """
    This endpoint receives a video and an image and returns a video with the face swapped.
    """
    if "video" not in request.files or "image" not in request.files:
        return jsonify({"error": "video and image are required"}), 400

    video = request.files["video"]
    image = request.files["image"]

    # Save the uploaded files
    video_filename = secure_filename(video.filename)
    image_filename = secure_filename(image.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
    video.save(video_path)
    image.save(image_path)

    # Extract faces
    extract_command = [
        "python", app.config["FACESWAP_PATH"], "extract",
        "-i", app.config["UPLOAD_FOLDER"],
        "-o", os.path.join(app.config["PROCESSED_FOLDER"], "faces")
    ]
    subprocess.run(extract_command)

    # Train the model
    train_command = [
        "python", app.config["FACESWAP_PATH"], "train",
        "-A", os.path.join(app.config["PROCESSED_FOLDER"], "faces", "video"),
        "-B", os.path.join(app.config["PROCESSED_FOLDER"], "faces", "image"),
        "-m", os.path.join(app.config["PROCESSED_FOLDER"], "model")
    ]
    subprocess.run(train_command)

    # Convert the video
    convert_command = [
        "python", app.config["FACESWAP_PATH"], "convert",
        "-i", app.config["UPLOAD_FOLDER"],
        "-o", app.config["PROCESSED_FOLDER"],
        "-m", os.path.join(app.config["PROCESSED_FOLDER"], "model")
    ]
    subprocess.run(convert_command)

    processed_video_path = os.path.join(app.config["PROCESSED_FOLDER"], video_filename)
    return send_file(processed_video_path, mimetype="video/mp4")

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
