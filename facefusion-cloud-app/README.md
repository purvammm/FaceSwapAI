# FaceFusion Cloud App

This application allows you to upload a video and an image to a cloud storage service, and then (theoretically) trigger a cloud-based processing job to perform a face swap.

## Architecture

The application is divided into two main components:

-   **Frontend**: A simple HTML page with a form to upload the video and image.
-   **Backend**: A Flask application that handles the file uploads and interaction with the cloud storage service.

## Setup

1.  **Configure Cloud Storage:**
    -   Open `backend/app.py`.
    -   Replace `"your-s3-bucket-name"` and `"your-s3-region"` with your actual AWS S3 bucket name and region.
    -   If you haven't configured your AWS credentials in your environment, you can add them directly to the `boto3.client` call in `backend/app.py`.

2.  **Install Dependencies:**
    ```bash
    pip install -r backend/requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    python backend/app.py
    ```

4.  **Access the Frontend:**
    -   Open your web browser and go to `http://127.0.0.1:5000/`.

## How It Works

1.  The user selects a video and an image using the form on the frontend.
2.  The files are sent to the `/api/upload` endpoint on the backend.
3.  The backend uploads the files to the configured cloud storage bucket (e.g., AWS S3).
4.  The backend then (in a real application) triggers a cloud processing job (e.g., an AWS Lambda function) to perform the face swap.
5.  The frontend receives a message indicating that the files have been uploaded and processing has started.

## Next Steps

To make this a fully functional application, you would need to:

1.  **Implement the Cloud Processing Job:**
    -   Create a serverless function or a containerized job that uses a face-swapping library (like FaceFusion) to process the video.
    -   This job would download the video and image from cloud storage, perform the face swap, and upload the resulting video back to cloud storage.
2.  **Implement Status Polling:**
    -   Add an endpoint to the backend (e.g., `/api/status/<job_id>`) that the frontend can poll to check the status of the processing job.
3.  **Display the Result:**
    -   Once the processing is complete, the frontend would receive the URL of the processed video and display it to the user.
