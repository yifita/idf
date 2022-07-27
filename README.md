# Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields
[[project page](https://yifita.github.io/publication/idf/)][[paper](http://arxiv.org/abs/2106.05187)][[cite](#bibtex)]

- [Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields](#geometry-consistent-neural-shape-representation-with-implicit-displacement-fields)
  - [overview](#overview)
  - [demos](#demos)
    - [preparations](#preparations)
    - [surface reconstruction](#surface-reconstruction)
    - [detail transfer](#detail-transfer)
  - [bibtex](#bibtex)

## overview
[![overview video](https://img.youtube.com/vi/fl4Rje8HM3I/0.jpg)](https://www.youtube.com/watch?v=fl4Rje8HM3I "play in youtube")

## demos
cuda 11.1 and pytorch 3.8
### preparations
```bash
git clone https://github.com/yifita/idf.git
cd idf

# conda environment and dependencies
# update conda
conda update -n base -c defaults conda
# install requirements
conda env create --name idf -f environment.yml
conda activate idf

# download data. This will download 8 mesh and point clouds to data/benchmark_shapes
sh data/get_data.sh
```
### surface reconstruction
```bash
# surface reconstruction from point cloud
# replace {asian_dragon} with another model name inside the benchmark_shape folder
python net/classes/runner.py net/experiments/displacement_benchmark/ablation/ablation_phased_scaledTanh_yes_act_yes_baseLoss_yes.json --name asian_dragon

```

### detail transfer
This example uses provided base shapes
```bash
sh data/get_dt_shapes.sh

# evaluation of the pretrained examples. This will save the results in 'runs/shorts_residual_filmsiren'
python net/classes/runner.py net/experiments/transfer/shorts_2phase.json
```

Or you could also train these examples yourselves:
```bash

sh data/get_dt_shapes.sh

# this will train the base shapes for the source and target shapes, then train the transferable idf
python net/classes/executor.py net/experiments/transfer/exec.json
```
## bibtex
```
@inproceedings{yifan2021geometry,
  title={Geometry-Consistent Neural Shape Representation with Implicit Displacement Fields},
  author={Yifan, Wang and Rahmann, Lukas and Sorkine-hornung, Olga},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```