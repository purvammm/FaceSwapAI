# FaceFusion Cloud App

This application allows you to upload a video and an image to a cloud storage service, and then trigger a cloud-based processing job to perform a face swap.

## Architecture

The application is divided into three main components:

-   **Frontend**: A simple HTML page with a form to upload the video and image.
-   **Backend**: A Flask application that handles the file uploads and interaction with the cloud storage service and the Lambda function.
-   **Lambda Function**: an AWS Lambda function that performs the face-swapping.

## Setup

### 1. Deploy the Lambda Function

1.  **Create a Lambda Function:**
    -   Go to the AWS Lambda console and create a new function.
    -   Choose "Author from scratch".
    -   Function name: `facefusion-processor` (or any other name you prefer).
    -   Runtime: Python 3.9.
    -   Architecture: x86_64.
2.  **Upload the Deployment Package:**
    -   In the "Code source" section, click on "Upload from" and select ".zip file".
    -   Upload the `lambda.zip` file.
3.  **Configure the Handler:**
    -   In the "Runtime settings" section, set the "Handler" to `src.main.handler`.
4.  **Increase the Timeout:**
    -   In the "Configuration" tab, go to "General configuration" and set the "Timeout" to a higher value, for example, 5 minutes.
5.  **Add a Trigger:**
    -   In the "Function overview" section, click on "Add trigger".
    -   Select "S3" as the trigger.
    -   Select your S3 bucket.
    -   Event type: "All object create events".
    -   Prefix (optional): `uploads/`.
    -   Suffix (optional): `.mp4`.
6.  **Add Permissions:**
    -   In the "Configuration" tab, go to "Permissions" and attach a policy to the execution role that allows the function to read from and write to your S3 bucket.

### 2. Configure the Backend

1.  **Open `backend/app.py`:**
    -   Replace `"your-s3-bucket-name"`, `"your-s3-region"`, and `"your-lambda-function-name"` with your actual values.
2.  **Install Dependencies:**
    ```bash
    pip install -r backend/requirements.txt
    ```

### 3. Run the Application

1.  **Run the Backend:**
    ```bash
    python backend/app.py
    ```
2.  **Access the Frontend:**
    -   Open your web browser and go to `http://127.0.0.1:5000/`.

## How It Works

1.  The user selects a video and an image using the form on the frontend.
2.  The files are sent to the `/api/upload` endpoint on the backend.
3.  The backend uploads the files to the configured S3 bucket.
4.  The upload to S3 triggers the Lambda function.
5.  The Lambda function downloads the video and image, performs the face swap, and uploads the processed video back to S3.
6.  The frontend polls the `/api/status` endpoint to check the status of the processing.
7.  Once the processing is complete, the frontend receives the URL of the processed video and displays it to the user.
