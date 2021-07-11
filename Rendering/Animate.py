import argparse
from Config import Config
from CameraConfiguration import CameraConfiguration
from Experiment import Experiment
from Mesh import Mesh
from PIL import Image
import bz2
import pickle
import _pickle as cPickle
import sys
import numpy as np
import pyrender
from Shader import ShaderCache
parser = argparse.ArgumentParser()
import trimesh

import time
#thai example ython ~/Thesis/resSDF/Rendering/Animate.py --mesh runs/Eval_thai_statue_GT/Eval/File/mesh_mesh_HighRes.ply --camera thai:animate --numsteps 200 --duration 10 --custom-shader --out gt_new_eval  --rotaxis 0 1 0 --target 0 0 -0.1 --height 1080

parser.add_argument(
        "--mesh",
        required=True,
        help="Mesh to animate"
)

parser.add_argument(
    "--camera",
    required=True,
    help="full camera name"
)

parser.add_argument(
    "--height",
    default=720,
    type=int,
    help="Height of the camera"
)


parser.add_argument(
    "--rotaxis",
    nargs=3,
    type=float,
    default=[0,0,0],
    help="Rotation axis, default = camera up axis"
)

parser.add_argument(
    "--target",
    nargs=3,
    type=float,
    default=[0,0,0],
    help="Rotation axis, default = camera up axis"
)

parser.add_argument(
    "--maxrotation",
    type = float,
    default=2*np.pi,
    help="Maximum rotation for loop leave on default"
)

parser.add_argument(
    "--numsteps",
    type = int,
    default=100,
    help="number of samples for the animation png"
)

parser.add_argument(
    "--duration",
    type = float,
    default=10,
    help="Theoretical rotation speed in seconds"
)



parser.add_argument(
        "--custom-shader",
        action="store_true",
        help="Uses the custom defined shader to run the script"        
)

parser.add_argument(
    "--out",
    required=True,
    help="Folder to save the results"
)


parser.add_argument(
    "--set-pose",
    action="store_true",
    help="Create a new pose/modify existing one for starting"
)

parser.add_argument(
    "--set-axis",
    action="store_true",
    help="Print and set a new rotation axis"
)


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, [1, 0, 0])
    >>> np.allclose(q, [0.99810947, 0.06146124, 0, 0])
    True

    """
    q = np.array([axis[0], axis[1], axis[2],0.0])
    qlen = np.linalg.norm(q)
    if qlen > 1e-6:
        q *= np.sin(angle / 2.0) / qlen
    q[3] = np.cos(angle / 2.0)
    return q



args = parser.parse_args()
if(args.rotaxis == [0, 0, 0 ]):
    print("Using Camera axis")
    args.rotaxis = None

all_images = {}
path_format = f"{args.out}/{{0}}"
Config.use_custom_shader = args.custom_shader

Config.camera_folder = args.out
CameraConfiguration.load_all()

model = Mesh("Animate",args.mesh,normalize=True)
target = args.target

if(args.set_axis):
    camera = CameraConfiguration()
    viewer = camera.get_viewer(args.height, model, viewer_flags = {"rotate":False, "windows_title":'select rotation axis axis'})
    rotate = False
    print(viewer.scene.scale)
    #ciewer._trackball._pose[:3,2] = args.rotaxis
    #viewer._trackball._pose[:3,3] = args.target

    while viewer._is_active:
        if not viewer.viewer_flags['rotate']:
            viewer.viewer_flags['rotate_axis'] = viewer._trackball._pose[:3,2]
            args.rotaxis = viewer._trackball._pose[:3,2]
            viewer.viewer_flags['rotate_rate'] = args.maxrotation/args.duration
            #print(args.rotaxis)
            rotate = False
            viewer._trackball._target = viewer._scene.centroid
            viewer._trackball._n_target = viewer._scene.centroid
            
        elif not rotate:
            rotate = True
            viewer._trackball._target = viewer._trackball._pose[:3,3]
            viewer._trackball._n_target = viewer._trackball._pose[:3,3]
            
           
        time.sleep(0.1)
    viewer.close_external()
    target = viewer._trackball._pose[:3,3]
    target -= target.dot(args.rotaxis) * args.rotaxis

print(f"Using axis --axis {args.rotaxis[0]} {args.rotaxis[1]} {args.rotaxis[2]} --target {target[0]} {target[1]} {target[2]}")

if(args.set_pose):
    def callback(i, v, scene, node):
        if(i==0):
            v.viewer_flags['rotate_axis'] = args.rotaxis
            v.viewer_flags['rotate_rate'] = args.maxrotation/args.duration
            v.viewer_flags['rotate']  = False
            v._trackball._target =target
            v._trackball._n_target =target
        time.sleep(0.5)
          
    camera = CameraConfiguration.set_camera(args.camera,model,args.height, callback)
    camera.export()
else:

    if(args.camera not in CameraConfiguration.configurations):
        print(f"Availible cameras : {CameraConfiguration.configurations.keys()}")
        sys.exit(-1)
    camera = CameraConfiguration.configurations[args.camera]
    #don't ask some buggy code open viewer and close to fix it
    viewer = camera.get_viewer(args.height, model, viewer_flags = {"rotate":True})
    viewer.close_external()



scene = camera.get_scene()
width = int(camera.ratio * args.height)


#attaching a light to the camera gives shadows
light = pyrender.DirectionalLight(color=Config.ambient_light, intensity=20.0)
light_node = pyrender.Node(light=light, matrix=np.eye(4))
scene.add_node(light_node,scene.main_camera_node)
model_node = model.get_node()
scene.add_node(model_node)


r = pyrender.OffscreenRenderer(viewport_width=width,
                    viewport_height=args.height,
                    point_size=1.0)

if Config.use_custom_shader:
        r._renderer._program_cache = ShaderCache()

from PIL import Image

if args.rotaxis  is None:
    args.rotaxis  = camera.pose[:3,1].flatten()

axis = np.array(args.rotaxis)
quat = np.zeros(4)
imgs = [] 
for i in range(args.numsteps+1):
    #scene.main_camera_node.rotation =np.array([0.0, 0.0, 0.0, 1.0]) # 
    angle = i/args.numsteps * args.maxrotation
    R = np.eye(4)
    R = trimesh.transformations.rotation_matrix(angle, args.rotaxis, target)
    scene.set_pose(scene.main_camera_node,R.dot(camera.pose))
    #scene.set_pose(model_node,R)
    imgs.append(Image.fromarray(r.render(scene,flags=pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.RGBA)[0]))


#for i,img in enumerate(imgs):
#    img.save(f"animation_{i+1}.png")

img, *imgs = imgs
#apng is better than gif
with open(f"{args.out}/{camera.name}_animated.png","wb") as fp_out:
    img.save(fp=fp_out, format='PNG', append_images=imgs,
            save_all=True, duration=int(1000*args.duration/args.numsteps), loop=0)


#image, _ = r.render(scene,flags=pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.RGBA)


