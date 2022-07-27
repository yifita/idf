# surface construction
mkdir -p data/benchmark_shapes
wget https://igl.ethz.ch/projects/idf/benchmark_shapes.zip -P data/benchmark_shapes
unzip data/benchmark_shapes/benchmark_shapes.zip -d data/benchmark_shapes
rm data/benchmark_shapes/benchmark_shapes.zip