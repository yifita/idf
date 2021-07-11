
import argparse
import os
import sys
import inspect

from CameraConfiguration import CameraConfiguration
from Config import Config
from Mesh import Mesh


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    default=None,
    help="Save folder for the camera default is the same as the model"
)
parser.add_argument(
    "--model",
    required=True,
    help="""Path of the model file to define the camera for"""
)

parser.add_argument(
    "--camera",
    default="normal",
    help="Name of for camera pose"
)

args = parser.parse_args()

folder = os.path.dirname(args.model)
if args.output:
    folder = args.output
Config.basefolder = folder

def getfilename(path:str)->str:
    return os.path.splitext(os.path.basename(path))[0]

base_name = getfilename(args.model)
groundTruth = Mesh(base_name,args.model)
groundTruth.load()
groundTruth.normalize()
name = f"{base_name}:{args.camera}"
CameraConfiguration.load_all()
camera = CameraConfiguration.set_camera(name,groundTruth)
camera.export()