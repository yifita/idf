from CameraConfiguration import CameraConfiguration
from Mesh import Mesh
from typing import List,Dict, Tuple
import numpy as np
import trimesh

class Experiment:
    """Describes the set of one experiment (e.g. different methods using the same groundTruth) and compares those with each other
    Attributes:
        name (str): name to use for this experiment in the generated tables/models
        evaluations List[Mesh]: resulting meshes of this experiment
        groundtruth Mesh: ground truth mesh of this experiment
        scenes : camera poses to render
    """
    name : str = None
    evaluations : List[Mesh] = None
    groundtruth : Mesh = None
    scenes : List[CameraConfiguration] = None

    def __init__(self, name : str , groundtruth : Mesh):
        self.evaluations : List[Mesh] = []
        self.scenes : List[CameraConfiguration] = []
        self.groundtruth : Mesh = groundtruth
        self.name : str = name

    def add_evaluation_mesh(self, mesh : Mesh):
        self.evaluations.append(mesh)

    def evaluate(self, samples = 1000000):
        self.groundtruth.load()
        for mesh in self.evaluations:
            mesh.load()
            print(f"Computing Chamfer for mesh {mesh.name} with gt {self.groundtruth.name}")
            mesh.chamfer(self.groundtruth,samples)

    def render(self, height : int = 720) -> Dict[str, np.ndarray]:
        res = {}
        for scene in self.scenes:
            self.groundtruth.load()
            res[f"{scene.name}_groundtruth"] = scene.render(self.groundtruth)
            for mesh in self.evaluations:
                mesh.load()
                res[f"{scene.name}_{mesh.name}"] = scene.render(mesh,height)
        return res

    def export(self, filter : List[str], path_format : str ):
        for mesh in self.evaluations:
            for name,tmesh in mesh.children.items():
                if name in filter:
                    path = path_format.format(f"{mesh.name}_{self.name}_{name}") + ".ply"
                    f = open(path, "wb")
                    if(tmesh.faces):
                        data = trimesh.exchange.ply.export_ply(tmesh, 'binary',vertex_normal=False)
                    else:
                        data = trimesh.exchange.ply.export_ply(tmesh, 'binary',vertex_normal=True)
                    f.write(data)
                    f.close()

    def get_table_collumn(self, filter : List[str], fmt="{:.2e}") -> List[str]:
        values = [self.name]
        element_len = len(filter)
        for mesh in self.evaluations:
            value = ["-"]*element_len
            print(mesh.values.keys())
            for key,fvalue in  mesh.values.items():
                if(key in filter):
                    value[filter.index(key)] = fmt.format(fvalue)
            values.append("/".join(value))
        return values

    def add_scenes(self):
        for name,scene in CameraConfiguration.configurations.items():
            if name.starts_with(f"{self.groundtruth.name}:"):
                self.scenes.append(scene)

    def create_scene_if_not(self, name):
        fullname = f"{self.groundtruth.name}:{name}"
       
        if fullname not in CameraConfiguration.configurations:
            CameraConfiguration.set_camera(fullname,self.groundtruth).export()
        self.scenes.append(CameraConfiguration.configurations[fullname])
