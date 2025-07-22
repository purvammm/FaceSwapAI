#!/usr/bin/env python3
"""
Media processing for faceswap
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from lib.cli import FullPaths, SmartFormatter
from lib.multithreading import MultiThread


class Alignments():
    """
    Alignments file handling
    """
    def __init__(self, arguments):
        self.args = arguments
        self.folder = self.args.alignments_dir
        self.file = self.get_alignments_file()
        self.data = self.load()

    def get_alignments_file(self):
        """
        Get the alignments file path
        """
        if not self.folder:
            return None
        file = os.path.join(self.folder, "alignments.json")
        return file

    def load(self):
        """
        Load the alignments file
        """
        if not self.file or not os.path.exists(self.file):
            return dict()
        with open(self.file, "r") as f:
            data = json.load(f)
        return data

    def save(self):
        """
        Save the alignments file
        """
        if not self.file:
            return
        with open(self.file, "w") as f:
            json.dump(self.data, f)

    def bak(self):
        """
        Backup the alignments file
        """
        if not self.file or not os.path.exists(self.file):
            return
        bak_file = self.file + ".bak"
        if os.path.exists(bak_file):
            os.remove(bak_file)
        os.rename(self.file, bak_file)


class Faces():
    """
    Faces processing
    """
    def __init__(self, arguments):
        self.args = arguments
        self.input_dir = self.args.input_dir
        self.output_dir = self.args.output_dir
        self.detector = self.args.detector
        self.aligner = self.args.aligner
        self.masker = self.args.masker
        self.processes = self.args.processes
        self.alignments = Alignments(self.args)
        self.verify_output()

    def verify_output(self):
        """
        Verify that the output directory exists
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_faces(self):
        """
        Process the faces
        """
        # Get the images
        images = self.get_images()
        # Create the threads
        threads = list()
        for i in range(self.processes):
            thread = MultiThread(self.process_face, images)
            thread.start()
            threads.append(thread)
        # Wait for the threads to finish
        for thread in threads:
            thread.join()
        # Save the alignments
        self.alignments.save()

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

    def process_face(self, image_path):
        """
        Process a single face
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
            # Get the mask
            mask = self.get_mask(face, alignment)
            # Save the face
            self.save_face(image_path, i, face, alignment, mask)

    def get_faces(self, image):
        """
        Get the faces from the image
        """
        if self.detector == "hog":
            faces = self.get_faces_hog(image)
        elif self.detector == "cnn":
            faces = self.get_faces_cnn(image)
        elif self.detector == "mtcnn":
            faces = self.get_faces_mtcnn(image)
        return faces

    def get_faces_hog(self, image):
        """
        Get the faces from the image using the hog detector
        """
        import dlib
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 1)
        return [(r.left(), r.top(), r.width(), r.height()) for r in faces]

    def get_faces_cnn(self, image):
        """
        Get the faces from the image using the cnn detector
        """
        import dlib
        detector = dlib.cnn_face_detection_model_v1(
            "data/mmod_human_face_detector.dat")
        faces = detector(image, 1)
        return [(r.rect.left(), r.rect.top(), r.rect.width(),
                 r.rect.height()) for r in faces]

    def get_faces_mtcnn(self, image):
        """
        Get the faces from the image using the mtcnn detector
        """
        from mtcnn.mtcnn import MTCNN
        detector = MTCNN()
        faces = detector.detect_faces(image)
        return [f["box"] for f in faces]

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

    def get_mask(self, face, alignment):
        """
        Get the mask for the face
        """
        if self.masker == "vgg-clear":
            mask = self.get_mask_vgg_clear(face, alignment)
        elif self.masker == "vgg-obstructed":
            mask = self.get_mask_vgg_obstructed(face, alignment)
        elif self.masker == "unet-dfl":
            mask = self.get_mask_unet_dfl(face, alignment)
        return mask

    def get_mask_vgg_clear(self, face, alignment):
        """
        Get the mask for the face using the vgg-clear masker
        """
        # TODO
        return None

    def get_mask_vgg_obstructed(self, face, alignment):
        """
        Get the mask for the face using the vgg-obstructed masker
        """
        # TODO
        return None

    def get_mask_unet_dfl(self, face, alignment):
        """
        Get the mask for the face using the unet-dfl masker
        """
        # TODO
        return None

    def save_face(self, image_path, i, face, alignment, mask):
        """
        Save the face to the output directory
        """
        # Get the filename
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        # Save the face
        face_filename = os.path.join(self.output_dir,
                                     "{}_{}{}".format(name, i, ext))
        cv2.imwrite(face_filename, face)
        # Save the alignment
        self.alignments.data[face_filename] = {
            "alignment": alignment,
            "mask": mask
        }


class Frames():
    """
    Frames processing
    """
    def __init__(self, arguments):
        self.args = arguments
        self.input_dir = self.args.input_dir
        self.output_dir = self.args.output_dir
        self.save_frames = self.args.save_frames
        self.verify_output()

    def verify_output(self):
        """
        Verify that the output directory exists
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_frames(self):
        """
        Process the frames
        """
        # Get the videos
        videos = self.get_videos()
        # Process each video
        for video_path in videos:
            self.process_video(video_path)

    def get_videos(self):
        """
        Get the videos from the input directory
        """
        videos = list()
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith((".mp4", ".avi", ".mov")):
                    videos.append(os.path.join(root, file))
        return videos

    def process_video(self, video_path):
        """
        Process a single video
        """
        # Get the video capture
        cap = cv2.VideoCapture(video_path)
        # Get the frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Get the filename
        filename = os.path.basename(video_path)
        name, _ = os.path.splitext(filename)
        # Process each frame
        for i in tqdm(range(frame_count), desc="Processing frames"):
            # Get the frame
            ret, frame = cap.read()
            if not ret:
                break
            # Save the frame
            if self.save_frames:
                frame_filename = os.path.join(
                    self.output_dir, "{}_{}.png".format(name, i))
                cv2.imwrite(frame_filename, frame)
        # Release the video capture
        cap.release()
