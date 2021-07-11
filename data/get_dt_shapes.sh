mkdir -p data/dt_shapes
wget https://igl.ethz.ch/projects/idf/dt_shapes.zip -P data/dt_shapes
unzip data/dt_shapes/dt_shapes.zip -d data/dt_shapes
rm data/dt_shapes/dt_shapes.zip
# get pretrained base models
mkdir -p runs
wget https://igl.ethz.ch/projects/idf/pretrained.zip -P runs
unzip runs/pretrained.zip -d runs
rm runs/pretrained.zip