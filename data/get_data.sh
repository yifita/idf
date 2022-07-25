# surface construction
mkdir -p data/benchmark_shapes
wget https://igl.ethz.ch/projects/idf/benchmark_shapes.zip -P data/ # benchmark_shapes
unzip data/benchmark_shapes/benchmark_shapes.zip -d data/benchmark_shapes
rm data/benchmark_shapes/benchmark_shapes.zip

# get pretrained base models and sphere initializations
mkdir -p runs
wget https://igl.ethz.ch/projects/idf/pretrained.zip -P runs/
unzip runs/pretrained.zip -d runs
rm runs/pretrained.zip