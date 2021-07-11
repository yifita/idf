#!/bin/bash
module load gcc/6.3.0 python_gpu/3.8.5
source ~/.bashrc
DIR=$(dirname "$0")
python $DIR/net/classes/executor.py "$@"
