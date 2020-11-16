#!/usr/bin/env bash
set -euo pipefail
source functions.sh
check_data_root

mkdir -p "$DATA_ROOT/mpii"
cd "$DATA_ROOT/mpii" || exit 1

wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip

tar -xvf mpii_human_pose_v1.tar.gz
rm mpii_human_pose_v1.tar.gz
unzip -j mpii_human_pose_v1_u12_2.zip
rm mpii_human_pose_v1_u12_2.zip