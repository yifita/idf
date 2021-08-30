import argparse
import glob
import re
import os

import pymeshlab
import trimesh
import glob
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from typing import Dict,Union
from pykdtree.kdtree import KDTree
import sys
folder = None
normalize = None

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

def load_point_cloud(path,num_points = -1):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()
    coords = mesh.vertex_matrix().astype('float32')
    normals = mesh.vertex_normal_matrix().astype('float32')
    if num_points > 0  and num_points < coords.shape[0]:
        idx = np.random.permutation(coords.shape[0])[:num_points]
        coords = np.ascontiguousarray(coords[idx])
        normals = np.ascontiguousarray(normals[idx])
        print(coords.shape)

    return coords, normals

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

def log_scalar(name, scalar):
    print(f"{name}:{scalar}")

def save_pc(filename, mesh):
    save_dir=os.path.dirname(os.path.abspath(filename))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    test = mesh.visual.kind
    #there is a bug in trimesh of it only saving normals when we tell the exporter explicitly to do so for point clouds.
    #thus we are calling the exporter directly instead of mesh.export(...)
    f = open(filename, "wb")
    data = trimesh.exchange.ply.export_ply(mesh, 'binary',vertex_normal=True)
    f.write(data)
    f.close()


def Normalize(mesh):
    coords = mesh.vertices
    coord_max = np.max(coords, axis=0, keepdims=True)[0]
    coord_min = np.min(coords, axis=0, keepdims=True)[0]
    coord_center = 0.5*(coord_max + coord_min)
    coords -= coord_center
    scale = np.linalg.norm(coords,axis=1).max()
    coords /= scale

def Compute_Chamfer(
    mesheval : trimesh.Trimesh,
    pointcloudgroundtruth : trimesh.PointCloud,
    normalize = True,
    normalize_base = True
    ) -> Dict[str,Union[float,trimesh.Trimesh]]:

    normals = pointcloudgroundtruth.vertex_normals
    if(normalize):
        Normalize(mesheval)
        Normalize(pointcloudgroundtruth)
    elif(normalize_base):
        Normalize(pointcloudgroundtruth)
    #coords,idx = trimesh.sample.sample_surface_even(meshgt,numpoints)
    #normals = meshgt.face_normals[idx,:]
    coords = pointcloudgroundtruth.vertices
    coords_new,idx_new = trimesh.sample.sample_surface_even(mesheval,coords.shape[0])
    normals_new = mesheval.face_normals[idx_new,:]
    print("Mean:")
    print(coords_new.mean(axis=0), coords.mean(axis=0))
    print(f"Using {coords.shape[0]} samples")
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
    d1_max = d1.max()
    d2_max = d2.max()

    nd1_max = nd1.max()
    nd2_max = nd2.max()
    output = {}
    output[e2gt] = d1.mean()
    output[gt2e] = d2.mean()
    output[f"{e2gt}_max"] = d1_max
    output[f"{gt2e}_max"] = d2_max
    output["hausdorff"] = max(d1_max,d2_max)
    output["sum"] = (d1.mean()+d2.mean())
    output[f"{e2gt}_normal"] = nd1.mean()
    output[f"{gt2e}_normal"] = nd2.mean()
    output["hausdorff_normal"] = max(nd1_max,nd2_max)
    output["sum_normal"] = nd1.mean() + nd2.mean()
    output[f"{gt2e}_pc"] = trimesh.Trimesh(vertices=coords_new,
                                           vertex_colors=get_color(d1,"cool",d1.min(),d1.max()),
                                           normals = normals_new
                                           )
    output[f"{e2gt}_pc"] = trimesh.Trimesh(vertices=coords,
                                           vertex_colors=get_color(d2,"cool",d2.min(),d2.max()),
                                           normals = normals
                                           )
    output[f"{gt2e}_pc_normal"] = trimesh.Trimesh(vertices=coords_new,
                                           vertex_colors=get_color(nd1,"seismic",0,1),
                                           normals = normals_new
                                           )
    output[f"{e2gt}_pc_normal"] = trimesh.Trimesh(vertices=coords,
                                           vertex_colors=get_color(nd2,"seismic",0,1),
                                           normals = normals)
    return output


def Save_Chamfer(
    mesheval : trimesh.Trimesh,
    pointcloudgroundtruth : trimesh.PointCloud,
    folder : str,
    name : str = "chamfer_{0}_{1}",
    save_mesh = True,
    normalize = True,
    normalize_base = True
    ):
    output = Compute_Chamfer(mesheval,pointcloudgroundtruth,normalize,normalize_base)
    for k,v in output:
        if isinstance(v, float):
            log_scalar(k,v)
        elif save_mesh:
            save_pc(f"{folder}/{k}.ply",v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        help="Folder for the ouput"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="""Format String for the input the first {0} will be replaced by the model name"""
    )
    parser.add_argument(
        "--models",
        default="data/benchmark_shapes/{0}_small_normalized.ply",
        help="""Format string for the model folder the first {0} will be replace\
                d by the name of the model"""
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="""Normalize the meshes before evaluating"""
    )

    parser.add_argument(
        "--normalize_base",
        action="store_true",
        help="""Normalize the meshes before evaluating"""
    )
    parser.add_argument(
        "--samples",
        default=4000000,type=int,
        help="""Number of points to evaluate"""
    )


    original_stdout = sys.stdout # Save a reference to the original standard output
    args = parser.parse_args()
    input_wildcard = args.input
    if( not os.path.isdir(args.output)):
        os.makedirs(args.output)
    std = Tee(f"{args.output}/chamfer.txt", 'w')
    regex = re.compile(input_wildcard.format("([A-z_]*)"))
    files = glob.glob(input_wildcard.format("[A-z_]*"))
    files.sort()
    for file in  files:
        print(file)

        match = regex.search(file)
        if(match):
            if(len(match.groups())==0):
                modelname = "static_model"
                modelfile = args.models
            else:
                modelname = match.group(1)
                modelfile = args.models.format(modelname)
            if(os.path.exists(modelfile)):
                print(f"\033[91m", file = original_stdout)
                print(f"{modelname}:")
                print(f"\033[0m", file = original_stdout)
                meshEval = trimesh.load(file, process=False)
                meshGt = load_point_cloud(modelfile,args.samples)
                print(f"loaded {meshGt[0].shape[0]} points", file = original_stdout)
                meshGt = trimesh.Trimesh(meshGt[0],vertex_normals=meshGt[1])
                Save_Chamfer(meshEval,meshGt,f"{args.output}/{modelname}_", normalize=args.normalize, normalize_base=args.normalize_base)
    del std
# hello
