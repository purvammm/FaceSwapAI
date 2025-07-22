#!/usr/bin/env python3
"""
Training script for faceswap
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from lib.cli import FullPaths, SmartFormatter
from lib.multithreading import MultiThread
from lib.model import Model


class Train():
    """
    Training session
    """
    def __init__(self, arguments):
        self.args = arguments
        self.input_A = self.args.input_A
        self.input_B = self.args.input_B
        self.model_dir = self.args.model_dir
        self.trainer = self.args.trainer
        self.save_interval = self.args.save_interval
        self.preview = self.args.preview
        self.write_image = self.args.write_image
        self.gpus = self.args.gpus
        self.batch_size = self.args.batch_size
        self.iterations = self.args.iterations
        self.preview_scale = self.args.preview_scale
        self.model = self.get_model()
        self.verify_output()

    def get_model(self):
        """
        Get the model
        """
        model = Model(self.model_dir, self.gpus)
        model.build(self.trainer)
        return model

    def verify_output(self):
        """
        Verify that the output directory exists
        """
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def process(self):
        """
        Process the training
        """
        # Get the images
        images_A = self.get_images(self.input_A)
        images_B = self.get_images(self.input_B)
        # Create the threads
        threads = list()
        for i in range(self.gpus):
            thread = MultiThread(self.train,
                                 (images_A, images_B, i))
            thread.start()
            threads.append(thread)
        # Wait for the threads to finish
        for thread in threads:
            thread.join()

    def get_images(self, input_dir):
        """
        Get the images from the input directory
        """
        images = list()
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(root, file))
        return images

    def train(self, images_A, images_B, gpu_id):
        """
        Train the model
        """
        # Set the GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Get the model
        model = self.get_model()
        # Get the training data
        data_A = self.get_data(images_A)
        data_B = self.get_data(images_B)
        # Train the model
        for i in tqdm(range(self.iterations), desc="Training"):
            # Get the batch
            batch_A = self.get_batch(data_A)
            batch_B = self.get_batch(data_B)
            # Train the model
            loss = model.train(batch_A, batch_B)
            # Save the model
            if i % self.save_interval == 0:
                model.save()
            # Show the preview
            if self.preview:
                self.show_preview(model, batch_A, batch_B)
            # Write the image
            if self.write_image and i % self.save_interval == 0:
                self.write_preview(model, batch_A, batch_B, i)

    def get_data(self, images):
        """
        Get the training data
        """
        data = list()
        for image_path in images:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64, 64))
            image = image.astype(np.float32) / 255.0
            data.append(image)
        return np.array(data)

    def get_batch(self, data):
        """
        Get a batch of training data
        """
        idx = np.random.randint(0, len(data), self.batch_size)
        return data[idx]

    def show_preview(self, model, batch_A, batch_B):
        """
        Show a preview of the training
        """
        # Get the predictions
        pred_A = model.predict(batch_A)
        pred_B = model.predict(batch_B)
        # Create the preview image
        preview = self.create_preview(batch_A, batch_B, pred_A, pred_B)
        # Show the preview
        cv2.imshow("preview", preview)
        cv2.waitKey(1)

    def write_preview(self, model, batch_A, batch_B, iteration):
        """
        Write a preview of the training to a file
        """
        # Get the predictions
        pred_A = model.predict(batch_A)
        pred_B = model.predict(batch_B)
        # Create the preview image
        preview = self.create_preview(batch_A, batch_B, pred_A, pred_B)
        # Write the preview
        filename = os.path.join(self.model_dir,
                                "preview_{}.png".format(iteration))
        cv2.imwrite(filename, preview)

    def create_preview(self, batch_A, batch_B, pred_A, pred_B):
        """
        Create a preview image
        """
        # Get the images
        img_A = self.get_image(batch_A)
        img_B = self.get_image(batch_B)
        img_pred_A = self.get_image(pred_A)
        img_pred_B = self.get_image(pred_B)
        # Create the preview
        preview = np.concatenate((img_A, img_pred_A, img_B, img_pred_B),
                                 axis=1)
        # Scale the preview
        preview = cv2.resize(preview, (0, 0), fx=self.preview_scale / 100,
                             fy=self.preview_scale / 100)
        return preview

    def get_image(self, batch):
        """
        Get an image from a batch
        """
        image = np.concatenate(batch, axis=1)
        image = (image * 255).astype(np.uint8)
        return image
