import argparse
from Config import Config
from CameraConfiguration import CameraConfiguration
from Experiment import Experiment
from Mesh import Mesh
from PIL import Image
import bz2
import pickle
import _pickle as cPickle
import os
#example --experiments 1:runs/Eval_{0}_Nglod/Eval/File/mesh_mesh_LOD6.ply --models camera asian_dragon --scenes normal close --out out_folder


def compressed_pickle(title, data):
    with bz2.BZ2File(title + ".pbz2", "w") as f:
        cPickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file+ ".pbz2", "rb")
    data = cPickle.load(data)
    return data


parser = argparse.ArgumentParser()

parser.add_argument(
        "--experiments",
        nargs="*",
        default=[],
        help="Mesh of the experiment to evaluate with {0} being a placeholder for the models"
    )

parser.add_argument(
        "--models",
        nargs="+",
        default=["asian_dragon","camera", "dragon","dragon_warrior","dragon_wing", "statue_ramesses", "thai_statue",  "vase_lion"],
        help="Name of the meshes to evaluate"
    )

parser.add_argument(
        "--modelpath",
        default="data/benchmark_shapes/{0}.off",
        help="Pat to meshes used for ground truth with {0} being a placeholder for the model name"
    )

parser.add_argument(
        "--scenes",
        nargs="+",
        default=[],
        help="Scenes to evaluate, opens dialog for every non defined scene/model"
    )

parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height in px of the resulting image"
    )

parser.add_argument(
        "--disable-normalize",
        action="store_false", dest="normalize",
        help="Normalize mesh before rendering"
    )

parser.add_argument(
        "--flipgt",
        action="store_true",
        help="Flip the gt meshes"
    )

parser.add_argument(
        "--samples",
        type=int,
        default=4000000,
        help="Number of samples to use when computing the chamfer distance"

)
parser.add_argument(
        "--chamfer",
        type=str,
        nargs="+",
        default= [],
        help="Evaluate and chamfer distance and save the value table possbile values are [sum,sum_normal,evalToGt,evalToGt_pc_normal,evalToGt_pc_normal,gtToEval,gtToEval_pc_normal,gtToEval_pc_normal] "
)

parser.add_argument(
        "--chamfer_mesh",
        type=str,
        nargs="+",
        default= [],
        help="Export chamfer meshes possible values are [evalToGt_pc, gtToEval_pc, evalToGt_pc_normal,gtToEval_pc_normal ]"
)

parser.add_argument(
        "--load",
        action="store_true",
        help="Disable chamfer evaluation only produce results rerun with same args in this case"
)

parser.add_argument(
        "--save",
        action="store_true",
        help="Serialize intermediate results for fast table creation etc"
)

parser.add_argument(
        "--custom-shader",
        action="store_true",
        help="Uses the custom defined shader to run the script"
)

parser.add_argument(
        "--camera-folder",
        default=None,
        help="Path to defined cameras default out direcotry"
    )



parser.add_argument(
    "--out",
    required=True,
    help="Folder to save the results"
)

args = parser.parse_args()
all_images = {}
path_format = f"{args.out}/{{0}}"
Config.use_custom_shader = args.custom_shader


if not args.camera_folder:
    args.camera_folder = args.out
if not args.load:
    experiments = []
    Config.camera_folder = args.camera_folder 
    CameraConfiguration.load_all()
    print(f"Setting up")
    for model in args.models:
        filemodel = args.modelpath.format(model)
        if not os.path.exists(filemodel):
            print(f"Skipping over missing modelfile {filemodel}")
            continue
        gT = Mesh(model,filemodel,normalize=True,flip=args.flipgt)
        experiment : Experiment = Experiment(model,gT)
        experiments.append(experiment)
        for experiment_arg in args.experiments:
            name, filename = experiment_arg.split(":",1)
            mesh = Mesh(name,filename.format(model), normalize=args.normalize)
            experiment.add_evaluation_mesh(mesh)

    for scene in args.scenes:
        for experiment in experiments:
            experiment.create_scene_if_not(scene)


    for experiment in experiments:
        if len(args.scenes)  > 0 :
            images = experiment.render(args.height)
            all_images.update(images)
        if len(args.chamfer) >0 or len(args.chamfer_mesh) > 0 :
            experiment.evaluate(args.samples)
            experiment.export(args.chamfer_mesh,path_format)
    if(args.save):
       compressed_pickle(f"{args.out}/chk.pkl",experiments)
else:
    experiments = decompress_pickle(f"{args.out}/chk.pkl")
    for experiment in experiments:
        if len(args.scenes)  > 0 :
            images = experiment.render(args.height)
            all_images.update(images)
        if len(args.chamfer) >0 or len(args.chamfer_mesh) > 0 :
            experiment.export(args.chamfer_mesh,path_format)

for k,v in all_images.items():
    im = Image.fromarray(v)
    im.save(f"{args.out}/{k}.png")

if len(args.chamfer) >0 :
    header = ["/".join(args.chamfer)]
    values = [[]]
    for experiment_arg in args.experiments:
       name, filename = experiment_arg.split(":",1)
       header.append(name)
    for experiment in experiments:
        values.append(experiment.get_table_collumn(args.chamfer))
    import tabulate
    print(header)
    print(values)
    results = tabulate.tabulate(values,headers=header,tablefmt="presto")
    print(results)
    f = open(f"{args.out}/results.txt","a")
    f.write(results)
    f.close

