#!/usr/bin/env python3
""" Command Line Arguments """
import argparse
import gettext
import sys
import textwrap

from . import __version__

# LOCALES
_LANG = gettext.translation("faceswap", localedir="locales", fallback=True)
_ = _LANG.gettext

# Python version check
if sys.version_info[0] < 3:
    raise Exception("This program requires at least python3.2")
if sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception("This program requires at least python3.2")


class FaceSwapArgs():
    """ Faceswap argument parser """
    def __init__(self, subparser=False):
        self.parser = self.get_parser(subparser)
        self.add_arguments()

    @staticmethod
    def get_parser(subparser):
        """ The parser for the tool """
        if subparser:
            parser = subparser.add_parser(
                "faceswap",
                description=_("Faceswap GUI."),
                epilog=_("For further help see: "
                         "https://github.com/deepfakes/faceswap-gui/"
                         "wiki/Help-and-Support"),
                conflict_handler="resolve",
                formatter_class=argparse.RawTextHelpFormatter)
            return parser

        parser = argparse.ArgumentParser(
            description=_("Faceswap.py - A tool for swapping faces in "
                          "videos and images"),
            epilog=_("For further help see: "
                     "https://github.com/deepfakes/faceswap/wiki"),
            conflict_handler="resolve",
            formatter_class=argparse.RawTextHelpFormatter)
        return parser

    def add_arguments(self):
        """ Add the arguments to the parser """
        self.parser.add_argument(
            "-v", "--version",
            action="version",
            version="Faceswap.py {}".format(__version__))

        subparser = self.parser.add_subparsers(
            dest="script",
            help=_("[WIP] Choose a command"))

        self.add_extract_arguments(subparser)
        self.add_train_arguments(subparser)
        self.add_convert_arguments(subparser)
        self.add_gui_arguments(subparser)

    @staticmethod
    def add_extract_arguments(subparser):
        """ Add the extract arguments to the parser """
        parser = subparser.add_parser(
            "extract",
            help=_("Extract faces from images or videos"),
            epilog=_("For further help see: "
                     "https://github.com/deepfakes/faceswap/wiki/Extraction"),
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument(
            "-i", "--input-dir",
            required=True,
            action=FullPaths,
            dest="input_dir",
            help=_("Input directory. A directory containing the files "
                   "you wish to process"))
        parser.add_argument(
            "-o", "--output-dir",
            required=True,
            action=FullPaths,
            dest="output_dir",
            help=_("Output directory. This is where the cropped faces will "
                   "be stored."))
        parser.add_argument(
            "-D", "--detector",
            type=str,
            choices=("hog", "cnn", "mtcnn"),
            default="hog",
            help=_("The face detector to use."))
        parser.add_argument(
            "-A", "--aligner",
            type=str,
            choices=("dlib", "cv2-dnn"),
            default="dlib",
            help=_("The face aligner to use."))
        parser.add_argument(
            "-M", "--masker",
            type=str,
            choices=("vgg-clear", "vgg-obstructed", "unet-dfl"),
            default="vgg-clear",
            help=_("The face masker to use."))
        parser.add_argument(
            "-r", "--rotate-images",
            action="store_true",
            help=_("If a face is not found, rotate the images to try to "
                   "find a face"))
        parser.add_argument(
            "-s", "--save-frames",
            action="store_true",
            help=_("Save the frames from a video. If you are extracting "
                   "from a video, this will save the frames in the output "
                   "directory."))
        parser.add_argument(
            "-L", "--log-level",
            type=str,
            choices=("INFO", "VERBOSE", "DEBUG", "TRACE"),
            default="INFO",
            help=_("The log level to use. (default: INFO)"))
        parser.add_argument(
            "-P", "--processes",
            type=int,
            default=1,
            help=_("The number of processes to use."))

    @staticmethod
    def add_train_arguments(subparser):
        """ Add the train arguments to the parser """
        parser = subparser.add_parser(
            "train",
            help=_("Train a model for the two faces A and B"),
            epilog=_("For further help see: "
                     "https://github.com/deepfakes/faceswap/wiki/Training"),
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument(
            "-A", "--input-A",
            required=True,
            action=FullPaths,
            dest="input_A",
            help=_("Input directory for face A. A directory containing "
                   "the cropped faces for face A."))
        parser.add_argument(
            "-B", "--input-B",
            required=True,
            action=FullPaths,
            dest="input_B",
            help=_("Input directory for face B. A directory containing "
                   "the cropped faces for face B."))
        parser.add_argument(
            "-m", "--model-dir",
            required=True,
            action=FullPaths,
            dest="model_dir",
            help=_("Model directory. This is where the model will be "
                   "stored."))
        parser.add_argument(
            "-t", "--trainer",
            type=str,
            choices=("original", "lowmem", "dfl-h128", "dfl-sae"),
            default="original",
            help=_("The trainer to use."))
        parser.add_argument(
            "-s", "--save-interval",
            type=int,
            default=100,
            help=_("The number of iterations between saving the model."))
        parser.add_argument(
            "-p", "--preview",
            action="store_true",
            help=_("Show a preview of the training process."))
        parser.add_argument(
            "-w", "--write-image",
            action="store_true",
            help=_("Writes the training result to a file every "
                   "save_interval"))
        parser.add_argument(
            "-L", "--log-level",
            type=str,
            choices=("INFO", "VERBOSE", "DEBUG", "TRACE"),
            default="INFO",
            help=_("The log level to use. (default: INFO)"))
        parser.add_argument(
            "-g", "--gpus",
            type=int,
            default=1,
            help=_("The number of GPUs to use."))
        parser.add_argument(
            "-bs", "--batch-size",
            type=int,
            default=64,
            help=_("The batch size to use."))
        parser.add_argument(
            "-it", "--iterations",
            type=int,
            default=1000000,
            help=_("The number of iterations to train for."))
        parser.add_argument(
            "-ps", "--preview-scale",
            type=int,
            default=100,
            help=_("The percentage to scale the preview by."))

    @staticmethod
    def add_convert_arguments(subparser):
        """ Add the convert arguments to the parser """
        parser = subparser.add_parser(
            "convert",
            help=_("Convert a video or images to a new video or images "
                   "with the face swapped"),
            epilog=_("For further help see: "
                     "https://github.com/deepfakes/faceswap/wiki/Conversion"),
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument(
            "-i", "--input-dir",
            required=True,
            action=FullPaths,
            dest="input_dir",
            help=_("Input directory. A directory containing the files "
                   "you wish to process"))
        parser.add_argument(
            "-o", "--output-dir",
            required=True,
            action=FullPaths,
            dest="output_dir",
            help=_("Output directory. This is where the converted files "
                   "will be stored."))
        parser.add_argument(
            "-m", "--model-dir",
            required=True,
            action=FullPaths,
            dest="model_dir",
            help=_("Model directory. This is where the model is "
                   "stored."))
        parser.add_argument(
            "-A", "--aligner",
            type=str,
            choices=("dlib", "cv2-dnn"),
            default="dlib",
            help=_("The face aligner to use."))
        parser.add_argument(
            "-c", "--converter",
            type=str,
            choices=("original", "mask", "mask-blend", "dfl-sae"),
            default="original",
            help=_("The converter to use."))
        parser.add_argument(
            "-b", "--blur-size",
            type=int,
            default=2,
            help=_("The blur size to use for the mask."))
        parser.add_argument(
            "-s", "--seamless-clone",
            action="store_true",
            help=_("Use seamless clone to blend the face."))
        parser.add_argument(
            "-M", "--mask-type",
            type=str,
            choices=("rect", "facehull", "vgg-clear", "vgg-obstructed",
                     "unet-dfl"),
            default="facehull",
            help=_("The mask type to use."))
        parser.add_argument(
            "-e", "--erosion-kernel-size",
            type=int,
            default=None,
            help=_("The erosion kernel size to use for the mask."))
        parser.add_argument(
            "-t", "--threshold",
            type=int,
            default=4,
            help=_("The threshold to use for the mask."))
        parser.add_argument(
            "-L", "--log-level",
            type=str,
            choices=("INFO", "VERBOSE", "DEBUG", "TRACE"),
            default="INFO",
            help=_("The log level to use. (default: INFO)"))
        parser.add_argument(
            "-P", "--processes",
            type=int,
            default=1,
            help=_("The number of processes to use."))
        parser.add_argument(
            "-g", "--gpus",
            type=int,
            default=1,
            help=_("The number of GPUs to use."))
        parser.add_argument(
            "-fr", "--frame-ranges",
            nargs="+",
            help=_("the frame ranges to apply conversion to. "
                   "eg. '1' '5-10'"))

    @staticmethod
    def add_gui_arguments(subparser):
        """ Add the train arguments to the parser """
        parser = subparser.add_parser(
            "gui",
            help=_("Launch the Faceswap GUI"),
            epilog=_("For further help see: "
                     "https://github.com/deepfakes/faceswap/wiki/GUI"),
            formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument(
            "-L", "--log-level",
            type=str,
            choices=("INFO", "VERBOSE", "DEBUG", "TRACE"),
            default="INFO",
            help=_("The log level to use. (default: INFO)"))

    def get_args(self):
        """ Get the arguments """
        return self.parser.parse_args()


class FullPaths(argparse.Action):
    """ Expand user- and relative-paths """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


class SmartFormatter(argparse.HelpFormatter):
    """ Smart Formatter for text wrapping """
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return argparse.HelpFormatter._split_lines(self, text, width)
