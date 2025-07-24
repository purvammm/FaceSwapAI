from flask import Flask, request, jsonify
import os
import boto3  # Using AWS S3 as an example for cloud storage
import json

app = Flask(__name__)

# Replace with your actual cloud storage credentials and bucket name
app.config["S3_BUCKET_NAME"] = "your-s3-bucket-name"
app.config["S3_REGION"] = "your-s3-region"
app.config["LAMBDA_FUNCTION_NAME"] = "your-lambda-function-name"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["PROCESSED_FOLDER"] = "processed"

s3 = boto3.client(
    "s3",
    region_name=app.config["S3_REGION"],
    # Add your AWS access key and secret key here if not configured in the environment
    # aws_access_key_id="YOUR_AWS_ACCESS_KEY_ID",
    # aws_secret_access_key="YOUR_AWS_SECRET_ACCESS_KEY",
)
lambda_client = boto3.client(
    "lambda",
    region_name=app.config["S3_REGION"],
    # Add your AWS access key and secret key here if not configured in the environment
    # aws_access_key_id="YOUR_AWS_ACCESS_KEY_ID",
    # aws_secret_access_key="YOUR_AWS_SECRET_ACCESS_KEY",
)

@app.route("/api/upload", methods=["POST"])
def upload():
    """
    This endpoint receives a video and an image, uploads them to cloud storage,
    and triggers a processing job.
    """
    if "video" not in request.files or "image" not in request.files:
        return jsonify({"error": "video and image are required"}), 400

    video = request.files["video"]
    image = request.files["image"]

    # Upload files to S3
    try:
        video_filename = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
        image_filename = os.path.join(app.config["UPLOAD_FOLDER"], "image.jpg") # Use a fixed name for the image

        s3.upload_fileobj(video, app.config["S3_BUCKET_NAME"], video_filename)
        s3.upload_fileobj(image, app.config["S3_BUCKET_NAME"], image_filename)

        # Trigger the Lambda function
        lambda_payload = {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": app.config["S3_BUCKET_NAME"]},
                        "object": {"key": video_filename},
                    }
                }
            ]
        }
        lambda_client.invoke(
            FunctionName=app.config["LAMBDA_FUNCTION_NAME"],
            InvocationType="Event",  # Use "Event" for asynchronous invocation
            Payload=json.dumps(lambda_payload),
        )

        return jsonify({
            "message": "Files uploaded successfully. Processing started.",
            "video_key": video_filename,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/status/<path:video_key>", methods=["GET"])
def status(video_key):
    """
    This endpoint checks the status of the video processing.
    """
    try:
        processed_key = os.path.join(app.config["PROCESSED_FOLDER"], os.path.basename(video_key))
        s3.head_object(Bucket=app.config["S3_BUCKET_NAME"], Key=processed_key)

        # If the object exists, the processing is complete
        # Generate a presigned URL for the processed video
        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": app.config["S3_BUCKET_NAME"], "Key": processed_key},
            ExpiresIn=3600,  # URL expires in 1 hour
        )
        return jsonify({"status": "complete", "url": presigned_url})

    except Exception as e:
        # If the object does not exist, the processing is still in progress
        return jsonify({"status": "processing"})

if __name__ == "__main__":
    app.run(debug=True)
