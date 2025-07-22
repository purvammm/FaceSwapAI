#!/usr/bin/env python3
"""
Conversion script for faceswap
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from lib.cli import FullPaths, SmartFormatter
from lib.multithreading import MultiThread
from lib.model import Model


class Convert():
    """
    Conversion session
    """
    def __init__(self, arguments):
        self.args = arguments
        self.input_dir = self.args.input_dir
        self.output_dir = self.args.output_dir
        self.model_dir = self.args.model_dir
        self.aligner = self.args.aligner
        self.converter = self.args.converter
        self.blur_size = self.args.blur_size
        self.seamless_clone = self.args.seamless_clone
        self.mask_type = self.args.mask_type
        self.erosion_kernel_size = self.args.erosion_kernel_size
        self.threshold = self.args.threshold
        self.processes = self.args.processes
        self.gpus = self.args.gpus
        self.frame_ranges = self.args.frame_ranges
        self.model = self.get_model()
        self.verify_output()

    def get_model(self):
        """
        Get the model
        """
        model = Model(self.model_dir, self.gpus)
        model.build()
        return model

    def verify_output(self):
        """
        Verify that the output directory exists
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process(self):
        """
        Process the conversion
        """
        # Get the images
        images = self.get_images()
        # Create the threads
        threads = list()
        for i in range(self.processes):
            thread = MultiThread(self.convert, images)
            thread.start()
            threads.append(thread)
        # Wait for the threads to finish
        for thread in threads:
            thread.join()

    def get_images(self):
        """
        Get the images from the input directory
        """
        images = list()
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append(os.path.join(root, file))
        return images

    def convert(self, image_path):
        """
        Convert a single image
        """
        # Get the image
        image = cv2.imread(image_path)
        # Get the faces
        faces = self.get_faces(image)
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Get the face
            face = image[y:y + h, x:x + w]
            # Get the alignment
            alignment = self.get_alignment(face)
            # Get the new face
            new_face = self.get_new_face(face)
            # Get the mask
            mask = self.get_mask(face, alignment)
            # Blend the face
            image[y:y + h, x:x + w] = self.blend(
                face, new_face, mask)
        # Save the image
        self.save_image(image_path, image)

    def get_faces(self, image):
        """
        Get the faces from the image
        """
        if self.aligner == "dlib":
            faces = self.get_faces_dlib(image)
        elif self.aligner == "cv2-dnn":
            faces = self.get_faces_cv2_dnn(image)
        return faces

    def get_faces_dlib(self, image):
        """
        Get the faces from the image using dlib
        """
        import dlib
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 1)
        return [(r.left(), r.top(), r.width(), r.height()) for r in faces]

    def get_faces_cv2_dnn(self, image):
        """
        Get the faces from the image using cv2-dnn
        """
        # TODO
        return None

    def get_alignment(self, face):
        """
        Get the alignment for the face
        """
        if self.aligner == "dlib":
            alignment = self.get_alignment_dlib(face)
        elif self.aligner == "cv2-dnn":
            alignment = self.get_alignment_cv2_dnn(face)
        return alignment

    def get_alignment_dlib(self, face):
        """
        Get the alignment for the face using dlib
        """
        import dlib
        predictor = dlib.shape_predictor(
            "data/shape_predictor_68_face_landmarks.dat")
        alignment = predictor(face, dlib.rectangle(
            0, 0, face.shape[1], face.shape[0]))
        return [(p.x, p.y) for p in alignment.parts()]

    def get_alignment_cv2_dnn(self, face):
        """
        Get the alignment for the face using cv2-dnn
        """
        # TODO
        return None

    def get_new_face(self, face):
        """
        Get the new face from the model
        """
        face = cv2.resize(face, (64, 64))
        face = face.astype(np.float32) / 255.0
        face = np.expand_dims(face, axis=0)
        new_face = self.model.predict(face)
        new_face = np.squeeze(new_face, axis=0)
        new_face = (new_face * 255).astype(np.uint8)
        return new_face

    def get_mask(self, face, alignment):
        """
        Get the mask for the face
        """
        if self.mask_type == "rect":
            mask = self.get_mask_rect(face)
        elif self.mask_type == "facehull":
            mask = self.get_mask_facehull(face, alignment)
        elif self.mask_type == "vgg-clear":
            mask = self.get_mask_vgg_clear(face, alignment)
        elif self.mask_type == "vgg-obstructed":
            mask = self.get_mask_vgg_obstructed(face, alignment)
        elif self.mask_type == "unet-dfl":
            mask = self.get_mask_unet_dfl(face, alignment)
        return mask

    def get_mask_rect(self, face):
        """
        Get a rectangular mask
        """
        mask = np.full(face.shape, 255, dtype=np.uint8)
        return mask

    def get_mask_facehull(self, face, alignment):
        """
        Get a facehull mask
        """
        mask = np.zeros(face.shape, dtype=np.uint8)
        points = np.array(alignment, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
        return mask

    def get_mask_vgg_clear(self, face, alignment):
        """
        Get a vgg-clear mask
        """
        # TODO
        return None

    def get_mask_vgg_obstructed(self, face, alignment):
        """
        Get a vgg-obstructed mask
        """
        # TODO
        return None

    def get_mask_unet_dfl(self, face, alignment):
        """
        Get a unet-dfl mask
        """
        # TODO
        return None

    def blend(self, face, new_face, mask):
        """
        Blend the new face onto the original face
        """
        if self.seamless_clone:
            return self.blend_seamless(face, new_face, mask)
        else:
            return self.blend_alpha(face, new_face, mask)

    def blend_alpha(self, face, new_face, mask):
        """
        Blend the new face onto the original face using alpha blending
        """
        mask = cv2.GaussianBlur(mask, (self.blur_size, self.blur_size), 0)
        mask = mask.astype(np.float32) / 255.0
        return (new_face * mask + face * (1 - mask)).astype(np.uint8)

    def blend_seamless(self, face, new_face, mask):
        """
        Blend the new face onto the original face using seamless cloning
        """
        center = (face.shape[1] // 2, face.shape[0] // 2)
        return cv2.seamlessClone(new_face, face, mask, center,
                                   cv2.NORMAL_CLONE)

    def save_image(self, image_path, image):
        """
        Save the image to the output directory
        """
        filename = os.path.basename(image_path)
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, image)
