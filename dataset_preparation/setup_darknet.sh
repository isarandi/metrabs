#!/usr/bin/env bash
set -euo pipefail
source functions.sh
check_data_root

git clone https://github.com/isarandi/darknet
cd darknet || exit 1

if [[ ! -d $CUDA_ROOT ]]; then
  echo "$CUDA_ROOT is not a directory! Set the \$CUDA_ROOT environment variable to the path of CUDA (e.g. /usr/local/cuda-10.0)"
  exit 1
fi

if [[ ! -w $CUDNN_ROOT ]]; then
  echo "$CUDNN_ROOT is not a directory! Set the \$CUDNN_ROOT environment variable to the path of cuDNN (e.g. /whatever/cudnn_7.5)"
  exit 1
fi
make
wget https://pjreddie.com/media/files/yolov3-spp.weights
