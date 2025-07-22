#!/usr/bin/env python3
"""
GUI for faceswap
"""
import os
import sys
import tkinter as tk
from tkinter import ttk
from lib.cli import FaceSwapArgs


class Gui():
    """
    GUI for faceswap
    """
    def __init__(self, arguments):
        self.args = arguments
        self.root = tk.Tk()
        self.root.title("Faceswap")
        self.create_widgets()

    def create_widgets(self):
        """
        Create the widgets for the GUI
        """
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both")
        self.create_extract_tab()
        self.create_train_tab()
        self.create_convert_tab()

    def create_extract_tab(self):
        """
        Create the extract tab
        """
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Extract")
        # Input directory
        tk.Label(tab, text="Input directory:").grid(
            row=0, column=0, sticky="w")
        self.extract_input_dir = tk.Entry(tab)
        self.extract_input_dir.grid(row=0, column=1, sticky="we")
        # Output directory
        tk.Label(tab, text="Output directory:").grid(
            row=1, column=0, sticky="w")
        self.extract_output_dir = tk.Entry(tab)
        self.extract_output_dir.grid(row=1, column=1, sticky="we")
        # Detector
        tk.Label(tab, text="Detector:").grid(row=2, column=0, sticky="w")
        self.extract_detector = ttk.Combobox(
            tab, values=("hog", "cnn", "mtcnn"))
        self.extract_detector.grid(row=2, column=1, sticky="we")
        self.extract_detector.set("hog")
        # Aligner
        tk.Label(tab, text="Aligner:").grid(row=3, column=0, sticky="w")
        self.extract_aligner = ttk.Combobox(
            tab, values=("dlib", "cv2-dnn"))
        self.extract_aligner.grid(row=3, column=1, sticky="we")
        self.extract_aligner.set("dlib")
        # Masker
        tk.Label(tab, text="Masker:").grid(row=4, column=0, sticky="w")
        self.extract_masker = ttk.Combobox(
            tab, values=("vgg-clear", "vgg-obstructed", "unet-dfl"))
        self.extract_masker.grid(row=4, column=1, sticky="we")
        self.extract_masker.set("vgg-clear")
        # Rotate images
        self.extract_rotate_images = tk.BooleanVar()
        tk.Checkbutton(tab, text="Rotate images",
                       variable=self.extract_rotate_images).grid(
                           row=5, column=0, sticky="w")
        # Save frames
        self.extract_save_frames = tk.BooleanVar()
        tk.Checkbutton(tab, text="Save frames",
                       variable=self.extract_save_frames).grid(
                           row=6, column=0, sticky="w")
        # Processes
        tk.Label(tab, text="Processes:").grid(row=7, column=0, sticky="w")
        self.extract_processes = tk.Entry(tab)
        self.extract_processes.grid(row=7, column=1, sticky="we")
        self.extract_processes.insert(0, "1")
        # Log level
        tk.Label(tab, text="Log level:").grid(row=8, column=0, sticky="w")
        self.extract_log_level = ttk.Combobox(
            tab, values=("INFO", "VERBOSE", "DEBUG", "TRACE"))
        self.extract_log_level.grid(row=8, column=1, sticky="we")
        self.extract_log_level.set("INFO")
        # Execute button
        tk.Button(tab, text="Extract",
                  command=self.execute_extract).grid(
                      row=9, column=0, columnspan=2)

    def create_train_tab(self):
        """
        Create the train tab
        """
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Train")
        # Input A
        tk.Label(tab, text="Input A:").grid(row=0, column=0, sticky="w")
        self.train_input_A = tk.Entry(tab)
        self.train_input_A.grid(row=0, column=1, sticky="we")
        # Input B
        tk.Label(tab, text="Input B:").grid(row=1, column=0, sticky="w")
        self.train_input_B = tk.Entry(tab)
        self.train_input_B.grid(row=1, column=1, sticky="we")
        # Model directory
        tk.Label(tab, text="Model directory:").grid(
            row=2, column=0, sticky="w")
        self.train_model_dir = tk.Entry(tab)
        self.train_model_dir.grid(row=2, column=1, sticky="we")
        # Trainer
        tk.Label(tab, text="Trainer:").grid(row=3, column=0, sticky="w")
        self.train_trainer = ttk.Combobox(
            tab, values=("original", "lowmem", "dfl-h128", "dfl-sae"))
        self.train_trainer.grid(row=3, column=1, sticky="we")
        self.train_trainer.set("original")
        # Save interval
        tk.Label(tab, text="Save interval:").grid(
            row=4, column=0, sticky="w")
        self.train_save_interval = tk.Entry(tab)
        self.train_save_interval.grid(row=4, column=1, sticky="we")
        self.train_save_interval.insert(0, "100")
        # Preview
        self.train_preview = tk.BooleanVar()
        tk.Checkbutton(tab, text="Preview",
                       variable=self.train_preview).grid(
                           row=5, column=0, sticky="w")
        # Write image
        self.train_write_image = tk.BooleanVar()
        tk.Checkbutton(tab, text="Write image",
                       variable=self.train_write_image).grid(
                           row=6, column=0, sticky="w")
        # GPUs
        tk.Label(tab, text="GPUs:").grid(row=7, column=0, sticky="w")
        self.train_gpus = tk.Entry(tab)
        self.train_gpus.grid(row=7, column=1, sticky="we")
        self.train_gpus.insert(0, "1")
        # Batch size
        tk.Label(tab, text="Batch size:").grid(row=8, column=0, sticky="w")
        self.train_batch_size = tk.Entry(tab)
        self.train_batch_size.grid(row=8, column=1, sticky="we")
        self.train_batch_size.insert(0, "64")
        # Iterations
        tk.Label(tab, text="Iterations:").grid(row=9, column=0, sticky="w")
        self.train_iterations = tk.Entry(tab)
        self.train_iterations.grid(row=9, column=1, sticky="we")
        self.train_iterations.insert(0, "1000000")
        # Preview scale
        tk.Label(tab, text="Preview scale:").grid(
            row=10, column=0, sticky="w")
        self.train_preview_scale = tk.Entry(tab)
        self.train_preview_scale.grid(row=10, column=1, sticky="we")
        self.train_preview_scale.insert(0, "100")
        # Log level
        tk.Label(tab, text="Log level:").grid(row=11, column=0, sticky="w")
        self.train_log_level = ttk.Combobox(
            tab, values=("INFO", "VERBOSE", "DEBUG", "TRACE"))
        self.train_log_level.grid(row=11, column=1, sticky="we")
        self.train_log_level.set("INFO")
        # Execute button
        tk.Button(tab, text="Train",
                  command=self.execute_train).grid(
                      row=12, column=0, columnspan=2)

    def create_convert_tab(self):
        """
        Create the convert tab
        """
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Convert")
        # Input directory
        tk.Label(tab, text="Input directory:").grid(
            row=0, column=0, sticky="w")
        self.convert_input_dir = tk.Entry(tab)
        self.convert_input_dir.grid(row=0, column=1, sticky="we")
        # Output directory
        tk.Label(tab, text="Output directory:").grid(
            row=1, column=0, sticky="w")
        self.convert_output_dir = tk.Entry(tab)
        self.convert_output_dir.grid(row=1, column=1, sticky="we")
        # Model directory
        tk.Label(tab, text="Model directory:").grid(
            row=2, column=0, sticky="w")
        self.convert_model_dir = tk.Entry(tab)
        self.convert_model_dir.grid(row=2, column=1, sticky="we")
        # Aligner
        tk.Label(tab, text="Aligner:").grid(row=3, column=0, sticky="w")
        self.convert_aligner = ttk.Combobox(
            tab, values=("dlib", "cv2-dnn"))
        self.convert_aligner.grid(row=3, column=1, sticky="we")
        self.convert_aligner.set("dlib")
        # Converter
        tk.Label(tab, text="Converter:").grid(row=4, column=0, sticky="w")
        self.convert_converter = ttk.Combobox(
            tab, values=("original", "mask", "mask-blend", "dfl-sae"))
        self.convert_converter.grid(row=4, column=1, sticky="we")
        self.convert_converter.set("original")
        # Blur size
        tk.Label(tab, text="Blur size:").grid(row=5, column=0, sticky="w")
        self.convert_blur_size = tk.Entry(tab)
        self.convert_blur_size.grid(row=5, column=1, sticky="we")
        self.convert_blur_size.insert(0, "2")
        # Seamless clone
        self.convert_seamless_clone = tk.BooleanVar()
        tk.Checkbutton(tab, text="Seamless clone",
                       variable=self.convert_seamless_clone).grid(
                           row=6, column=0, sticky="w")
        # Mask type
        tk.Label(tab, text="Mask type:").grid(row=7, column=0, sticky="w")
        self.convert_mask_type = ttk.Combobox(
            tab,
            values=("rect", "facehull", "vgg-clear", "vgg-obstructed",
                    "unet-dfl"))
        self.convert_mask_type.grid(row=7, column=1, sticky="we")
        self.convert_mask_type.set("facehull")
        # Erosion kernel size
        tk.Label(tab, text="Erosion kernel size:").grid(
            row=8, column=0, sticky="w")
        self.convert_erosion_kernel_size = tk.Entry(tab)
        self.convert_erosion_kernel_size.grid(row=8, column=1, sticky="we")
        # Threshold
        tk.Label(tab, text="Threshold:").grid(row=9, column=0, sticky="w")
        self.convert_threshold = tk.Entry(tab)
        self.convert_threshold.grid(row=9, column=1, sticky="we")
        self.convert_threshold.insert(0, "4")
        # Processes
        tk.Label(tab, text="Processes:").grid(
            row=10, column=0, sticky="w")
        self.convert_processes = tk.Entry(tab)
        self.convert_processes.grid(row=10, column=1, sticky="we")
        self.convert_processes.insert(0, "1")
        # GPUs
        tk.Label(tab, text="GPUs:").grid(row=11, column=0, sticky="w")
        self.convert_gpus = tk.Entry(tab)
        self.convert_gpus.grid(row=11, column=1, sticky="we")
        self.convert_gpus.insert(0, "1")
        # Frame ranges
        tk.Label(tab, text="Frame ranges:").grid(
            row=12, column=0, sticky="w")
        self.convert_frame_ranges = tk.Entry(tab)
        self.convert_frame_ranges.grid(row=12, column=1, sticky="we")
        # Log level
        tk.Label(tab, text="Log level:").grid(
            row=13, column=0, sticky="w")
        self.convert_log_level = ttk.Combobox(
            tab, values=("INFO", "VERBOSE", "DEBUG", "TRACE"))
        self.convert_log_level.grid(row=13, column=1, sticky="we")
        self.convert_log_level.set("INFO")
        # Execute button
        tk.Button(tab, text="Convert",
                  command=self.execute_convert).grid(
                      row=14, column=0, columnspan=2)

    def execute_extract(self):
        """
        Execute the extract command
        """
        args = ["extract"]
        args.extend(["-i", self.extract_input_dir.get()])
        args.extend(["-o", self.extract_output_dir.get()])
        args.extend(["-D", self.extract_detector.get()])
        args.extend(["-A", self.extract_aligner.get()])
        args.extend(["-M", self.extract_masker.get()])
        if self.extract_rotate_images.get():
            args.append("-r")
        if self.extract_save_frames.get():
            args.append("-s")
        args.extend(["-P", self.extract_processes.get()])
        args.extend(["-L", self.extract_log_level.get()])
        self.run(args)

    def execute_train(self):
        """
        Execute the train command
        """
        args = ["train"]
        args.extend(["-A", self.train_input_A.get()])
        args.extend(["-B", self.train_input_B.get()])
        args.extend(["-m", self.train_model_dir.get()])
        args.extend(["-t", self.train_trainer.get()])
        args.extend(["-s", self.train_save_interval.get()])
        if self.train_preview.get():
            args.append("-p")
        if self.train_write_image.get():
            args.append("-w")
        args.extend(["-g", self.train_gpus.get()])
        args.extend(["-bs", self.train_batch_size.get()])
        args.extend(["-it", self.train_iterations.get()])
        args.extend(["-ps", self.train_preview_scale.get()])
        args.extend(["-L", self.train_log_level.get()])
        self.run(args)

    def execute_convert(self):
        """
        Execute the convert command
        """
        args = ["convert"]
        args.extend(["-i", self.convert_input_dir.get()])
        args.extend(["-o", self.convert_output_dir.get()])
        args.extend(["-m", self.convert_model_dir.get()])
        args.extend(["-A", self.convert_aligner.get()])
        args.extend(["-c", self.convert_converter.get()])
        args.extend(["-b", self.convert_blur_size.get()])
        if self.convert_seamless_clone.get():
            args.append("-s")
        args.extend(["-M", self.convert_mask_type.get()])
        if self.convert_erosion_kernel_size.get():
            args.extend(["-e", self.convert_erosion_kernel_size.get()])
        args.extend(["-t", self.convert_threshold.get()])
        args.extend(["-P", self.convert_processes.get()])
        args.extend(["-g", self.convert_gpus.get()])
        if self.convert_frame_ranges.get():
            args.extend(["-fr", self.convert_frame_ranges.get()])
        args.extend(["-L", self.convert_log_level.get()])
        self.run(args)

    def run(self, args):
        """
        Run a command
        """
        command = [sys.executable, "faceswap.py"]
        command.extend(args)
        os.spawnv(os.P_NOWAIT, sys.executable, command)

    def process(self):
        """
        Process the GUI
        """
        self.root.mainloop()
