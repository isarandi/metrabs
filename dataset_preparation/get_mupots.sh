#!/usr/bin/env bash
set -euo pipefail
source functions.sh
check_data_root

mkdir -p "$DATA_ROOT/mupots"
cd "$DATA_ROOT/mupots" || exit 1

wget http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/content/mupots-3d-eval.zip
unzip mupots-3d-eval.zip
mv mupots-3d-eval/* ./
rmdir mupots-3d-eval
rm mupots-3d-eval.zip

wget http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/MultiPersonTestSet.zip
unzip MultiPersonTestSet.zip
mv MultiPersonTestSet/* ./
rmdir MultiPersonTestSet
rm MultiPersonTestSet.zip
