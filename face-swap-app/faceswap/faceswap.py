#!/usr/bin/env python3
""" The main entry point to the faceswap application """

# ENV CONFIG MUST BE AT THE TOP OF THE FILE
# pylint: disable=wrong-import-position
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import_fail = False
if sys.version_info[0] < 3:
    print("Faceswap requires Python 3.6 or later.")
    sys.exit(1)
if sys.version_info[1] < 6:
    print("Faceswap requires Python 3.6 or later.")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("OpenCV is not installed. Please install it with 'pip install opencv-python'")
    import_fail = True

try:
    import tensorflow
except ImportError:
    print("Tensorflow is not installed. Please install it with 'pip install tensorflow'")
    import_fail = True

if import_fail:
    sys.exit(1)

# TODO Move this check to be after argument parsing
if " ".join(sys.argv).startswith("python faceswap.py extract"):
    import tkinter
    if tkinter.TkVersion < 8.6:
        print("Tcl/Tk 8.6 is not installed. Please install it before running the GUI.")
        # TODO add more specific instructions
        sys.exit(1)

from lib.cli import FaceSwapArgs
from scripts.fsmedia import Alignments, Faces, Frames
from scripts.train import Train
from scripts.convert import Convert
from scripts.gui import Gui

if __name__ == "__main__":
    ARGUMENTS = FaceSwapArgs().get_args()

    if ARGUMENTS.script == "extract":
        FRAMES = Frames(ARGUMENTS)
        FACES = Faces(ARGUMENTS)
        Alignments(ARGUMENTS)
        FRAMES.process_frames()
        FACES.process_faces()
    elif ARGUMENTS.script == "train":
        Train(ARGUMENTS).process()
    elif ARGUMENTS.script == "convert":
        Convert(ARGUMENTS).process()
    elif ARGUMENTS.script == "gui":
        Gui(ARGUMENTS).process()
