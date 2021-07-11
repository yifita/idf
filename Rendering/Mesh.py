import pyrender
import numpy as np
import trimesh
from typing import List,Dict,Union
import sys
import os
import inspect
import PIL

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from computeChamfer import Compute_Chamfer

class Mesh(object):
    """A small wrapper for meshes to provide tools to compute the chamfer distance etc. and rendering over pyrenderer

    Attributes:
        filename (str): Filename used for importing/exporting the mesh
        translation (np.ndarray): Translation of the mesh e.g. for normalization
        scale (np.ndarray): scale of the mesh per axis
        children (List[Mesh]): list of the children meshes attributed with this mesh (e.g. chamfer distance)
        values : Dict[str,float]
    """
    filename : str = None
    mesh :  pyrender.mesh.Mesh = None
    name : str = None
    normalize_ : bool = False
    children : Dict[str,trimesh.Trimesh] = None
    values : Dict[str,float] = None

    def __init__(self, name : str,
                 filename : str = None,
                 vertices : np.ndarray = None,
                 faces : np.ndarray = None,
                 normals : np.ndarray = None,
                 colors : np.ndarray = None,
                 normalize : bool = False,
                 flip : bool = False
                 ):
        self.filename = filename
        self.name = name
        self.normalize_ = normalize
        self.children = {}
        self.values = {}
        self.flip = flip
        if vertices:
            if not faces:
                self.mesh = pyrender.Mesh.from_points(vertices,normals,colors)
            else:
                self.mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices,faces))
        if self.mesh and self.normalize_:
            self.normalize()


    def load(self):
        if not self.mesh:
            print(f"Loading mesh {self.filename}")
            mesh = trimesh.load_mesh(self.filename)
            if self.flip:
                mesh.faces = mesh.faces[:,::-1]
            self.mesh = pyrender.Mesh.from_trimesh(mesh,smooth = True)
            if self.normalize_:
                self.normalize()

    def chamfer(self, ground_truth : Union['Mesh',trimesh.PointCloud], count = 1000000, normalize = False, normalize_base = False):
        evaluation = trimesh.Trimesh(vertices = self.mesh.primitives[0].positions.copy(), faces=self.mesh.primitives[0].indices.copy())

        if( ground_truth.__class__ == self.__class__):
            ground_truth = trimesh.Trimesh(vertices = ground_truth.mesh.primitives[0].positions.copy(), faces=ground_truth.mesh.primitives[0].indices.copy())
            samples,idx = trimesh.sample.sample_surface(ground_truth,count)
            ground_truth = trimesh.Trimesh(vertices=samples,vertex_normals=ground_truth.face_normals[idx,:])
        else :
            return

        result = Compute_Chamfer(evaluation,ground_truth, normalize, normalize_base)
        for (k,v) in result.items():
            if(isinstance(v,float)):
                self.values[k] = v
            elif(isinstance(v,trimesh.Trimesh)):
                self.children[k] = v
            else:
                print(f"Stray value {v} with key {v} im chamfer evaluation")

    def normalize(self):
        assert(len(self.mesh.primitives) == 1)
        vertices = self.mesh.primitives[0].positions
        cordmax = vertices.max(axis=0)
        cordmin = vertices.min(axis=0)
        mean = (cordmax+cordmin)/2
        vertices -= mean
        vertices /= np.linalg.norm(vertices,axis=1).max()
        self.mesh.primitives[0].positions = vertices

    def show(self):
        pass

    def get_node(self):
        return pyrender.Node(name=self.name, mesh=self.mesh)