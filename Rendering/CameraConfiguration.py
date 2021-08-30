import glob
import pyrender
import os
from typing import Union,List,Dict,Tuple
import re
import numpy as np
from pickle import load,dump
from Config import Config
from Mesh import Mesh
from Shader import ShaderCache

class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(os.path.join(os.path.dirname(__file__),"shaders/mesh.vert"),
                                                                os.path.join(os.path.dirname(__file__),"shaders/mesh.frag"), defines=defines)
        return self.program

class CameraConfiguration:

    configurations : Dict[str,'CameraConfiguration'] = {}

    pose : np.ndarray = np.eye(4)
    yfov: float = np.pi / 3.0
    ratio: float = 16.0/9.0
    name : str = None

    @staticmethod
    def extract_from_viewer(viewer : pyrender.Viewer, config : "CameraConfiguration" = None) -> 'CameraConfiguration':
        if not config:
            config = CameraConfiguration()
        config.pose = viewer._camera_node.matrix.copy()
        config.yfov = viewer._camera_node.camera._yfov
        config.ratio = viewer.viewport_size[0]/viewer.viewport_size[1]
        return config

    @staticmethod
    def get_camera_file(name:str):
        path = os.path.abspath(Config.camera_folder)
        return f"{path}/camera_{name}.pkl"

    @staticmethod
    def load_all():
        g_path = CameraConfiguration.get_camera_file("*")
        g_regex = re.escape(CameraConfiguration.get_camera_file("REPLACEME")).replace("REPLACEME","([A-z\:]*)")
        pattern = re.compile(g_regex)
        for camera in glob.glob(g_path):
            try:
                match = pattern.search(camera)
                if(len(match.group())>0):
                    with open(camera, 'rb') as infile:
                        print(f"Loading Camera {match.group(1)}")
                        c = CameraConfiguration()
                        c.__dict__.update(load(infile))
                        CameraConfiguration.configurations[match.group(1)] = c
            except:
                pass

    @staticmethod
    def save_all():
        for camera in CameraConfiguration.configurations.values():
            camera.export()

    def get_viewer(self, height:int, mesh: Mesh, viewer_flags : Dict[str,object] = {} ) -> pyrender.Viewer:
        scene = self.get_scene()
        width = int(self.ratio * height)
        mesh.load()
        scene.add_node(mesh.get_node())
        viewer_flags["use_direct_lighting"]=True

        viewer = pyrender.Viewer(scene,viewport_size=(width,height),run_in_thread=True,
                                        render_flags={"all_solid":True},viewer_flags=viewer_flags)
        #this may not work correctly
        if Config.use_custom_shader:
            viewer._renderer._program_cache = ShaderCache()
        #while viewer.is_active:
        #    pass
        return viewer

    @staticmethod
    def set_camera(name : str, reference: Mesh, height: int = 720, callback = None) -> "CameraConfiguration":
        scene : pyrender.Scene = None
        # width : int = int(height * (16.0/9.0))
        width : int = height
        old_camera : "CameraConfiguration" = None

        if name in CameraConfiguration.configurations:
            camera = CameraConfiguration.configurations[name]
            scene = camera.get_scene()
            width = int(camera.ratio * height)
            old_camera = camera
        else:
            scene = Config.Scene()
        reference.load()
        meshnode = reference.get_node()
        scene.add_node(meshnode)

        viewer = pyrender.Viewer(scene,viewport_size=(width,height),run_in_thread=True,
                                        render_flags={"all_solid":True},viewer_flags={"use_direct_lighting":True,"window_title":f"Please Select a pose for {name}"})
        #this may not work correctly
        if Config.use_custom_shader:
            viewer._renderer._program_cache = ShaderCache()
        i = 0
        while viewer.is_active:
            if(callback is not None):
                callback(i,viewer, scene, meshnode)
            i+=1

        cam = CameraConfiguration.extract_from_viewer(viewer,old_camera)

        cam.name = name
        if cam not in CameraConfiguration.configurations:
            CameraConfiguration.configurations[name]=cam
        return cam

    def render(self, mesh:Mesh, height : int = 720) -> np.ndarray:
        scene = self.get_scene()
        width = int(self.ratio * height)

        r = pyrender.OffscreenRenderer(viewport_width=width,
                            viewport_height=height,
                            point_size=1.0)
        if Config.use_custom_shader:
                r._renderer._program_cache = ShaderCache()
        #attaching a light to the camera gives shadows
        light = pyrender.DirectionalLight(color=Config.ambient_light, intensity=20.0)
        light_node = pyrender.Node(light=light, matrix=np.eye(4))

        scene.add_node(light_node,scene.main_camera_node)
        scene.add_node(mesh.get_node())

        image, _ = r.render(scene,flags=pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.RGBA)

        return image

    def get_scene(self):
        scene = Config.Scene()
        cam = pyrender.PerspectiveCamera(yfov=self.yfov)
        cam_node = scene.add(cam, pose=self.pose)
        scene.main_camera_node = cam_node
        return scene

    def reload(self):
        path : str = CameraConfiguration.get_camera_file(self.name)
        with open(path, 'rb') as infile:
            self.__dict__.update(load(infile))
            print(f"Reloaded camera {self.name} filename {path}")


    def export(self):
        path : str = CameraConfiguration.get_camera_file(self.name)
        with open(path, 'wb') as outfile:
            dump(self.__dict__,outfile)
            print(f"Saved camera {self.name} filename {path}")


    def rm_camera(self, name:str):
        del self.configurations[name]