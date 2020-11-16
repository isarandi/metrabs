#!/usr/bin/env bash
set -euo pipefail
source functions.sh
check_data_root

mkdir -p "$DATA_ROOT/3dpw"
cd "$DATA_ROOT/3dpw" || exit 1

wget https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip
unzip imageFiles.zip
rm imageFiles.zip

wget https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip
unzip sequenceFiles.zip
rm sequenceFiles.zip
rm -rf __MACOSX