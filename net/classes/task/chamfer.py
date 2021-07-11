from pykdtree.kdtree import KDTree
from task.task import Task
import torch
from torch.utils.data import DataLoader
import trimesh
import glob
import numpy as np
import time
import os
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


def get_youngest_file(files):
    """ Find and return the oldest file of input file names.
    Only one wins tie. Values based on time distance from present.
    Use of `_invert` inverts logic to make this a youngest routine,
    to be used more clearly via `get_youngest_file`.
    """
     # Check for empty list.
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    # Raw epoch distance.
    now = time.time()
    # Select first as arbitrary sentinel file, storing name and age.
    youngest = files[0], now - os.path.getctime(files[0])
    # Iterate over all remaining files.
    for f in files[1:]:
        age = now - os.path.getctime(f)
        if age < youngest[1]:
            # Set new oldest.
            youngest = f, age
    # Return just the name of oldest file.
    return youngest[0]

def get_color(values, cmap="seismic",vmin = 0,vmax = 1):
    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    cMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    return cMap.to_rgba(values,bytes=True)[:,:3]


def NP_Cosine(v1,v2):
    d = np.sum(v1*v2,axis=1)
    n1 = np.linalg.norm(v1,axis=1)
    n2 = np.linalg.norm(v2,axis=1)

    sim =  (d/(n1*n2))
    #print(sim.min())
    return (1 - sim)/2

def Compute_Chamfer(mesh : trimesh.Trimesh,runner, name : str = "chamfer_{0}_{1}", save_mesh = True):
    coords = runner.data._coords.clone().detach().cpu().numpy().astype(np.float32)
    normals = runner.data._normals.clone().detach().cpu().numpy().astype(np.float32)
    coords_new,idx_new = trimesh.sample.sample_surface_even(mesh,coords.shape[0])
    coords_new = coords_new.astype(np.float32)
    normals_new = mesh.face_normals[idx_new,:]
    normals_new = normals_new.astype(np.float32)

    print(coords_new.dtype, coords.dtype)
    kdTree = KDTree(coords)
    d1,idx1 = kdTree.query(coords_new)
    #compute normal deformation
    nd1 = NP_Cosine(normals[idx1,:],normals_new)
    inverseTree = KDTree(coords_new)
    d2,idx2 = inverseTree.query(coords)
    #compute normal deformation
    nd2 = NP_Cosine(normals_new[idx2,:],normals)
    #mapping direction we either map
    e2gt = "evalToGt"
    gt2e = "gtToEval"

    runner.logger.log_hist(name.format(e2gt,"hist"),d1)
    runner.logger.log_hist(name.format(gt2e,"hist"),d2)

    runner.logger.log_hist(name.format(e2gt,"_normal_hist"),nd1)
    runner.logger.log_hist(name.format(e2gt,"_normal_hist"),nd2)
    runner.logger.log_scalar(name.format(e2gt,""),d1.mean())
    runner.logger.log_scalar(name.format(gt2e,""),d2.mean())
    runner.logger.log_scalar(name.format("sum",""),(d1+d2).mean())
    runner.logger.log_scalar(name.format(e2gt,"normal"),nd1.mean())
    runner.logger.log_scalar(name.format(gt2e,"normal"),nd2.mean())
    runner.logger.log_scalar(name.format("sum","normal"),(nd1+nd2).mean())
    runner.logger.log_scalar(name.format(e2gt,"median"),np.median(d1))
    runner.logger.log_scalar(name.format(gt2e,"median"),np.median(d2))

    if(save_mesh):
        runner.logger.log_mesh(name.format(e2gt,"normal_mesh"),
                            vertices = coords_new.reshape([1,-1,3]),
                            faces = None,
                            colors = get_color(d1,"cool",d1.min(),d1.max()).reshape(1,-1,3),
                            vertex_normals = normals_new.reshape(1,-1,3))
        runner.logger.log_mesh(name.format(gt2e,"normal_mesh"),
                            vertices = coords.reshape([1,-1,3]),
                            faces = None,
                            colors = get_color(d2,"cool",d2.min(),d2.max()).reshape(1,-1,3),
                            vertex_normals = normals.reshape(1,-1,3))

    #if(save_mesh):
    #    runner.logger.log_mesh(f"{folder}/{name.format(e2gt,'normal_mesh')}.ply",
    #                        vertices = coords_new,
    #                        colors = get_color(d1,"cool",d1.min(),d1.max()),
    #                        normals = normals_new)
    #    runner.logger.log_mesh(f"{folder}/{name.format(gt2e,'normal_mesh')}.ply",
    #                        vertices = coords,
    #                        colors = get_color(d2,"cool",d2.min(),d2.max()),
    #                        normals = normals)


class Chamfer(Task):

    def __init__(self, config):
        self.name_model : str = None
        self.tag : str = "HighRes"
        self.recenter : bool = False
        self.load_folder : str = None
        super().__init__(config)
        base_folder = os.path.join("runs/",self.runner.name)
        if(self.load_folder is None):
            self.load_folder = base_folder
        else:
            #relative folder
            self.load_folder = f"{base_folder}/{self.load_folder}"
        print(self.load_folder)
    def __call__(self):
        super().__call__()
        files = files = glob.glob(f"{self.load_folder}/*{self.tag}*")
        print(files)
        print(f"{self.load_folder}/*{self.tag}*")
        file = get_youngest_file(files)
        len(self.runner.data)

        self.runner.py_logger.info(f'Loading mesh {file}.')
        mesh = trimesh.load(file)
        Compute_Chamfer(mesh,self.runner)



