#!/usr/bin/env bash
set -euo pipefail
source functions.sh
check_data_root

mkdir -p "$DATA_ROOT/pascal_voc"
cd "$DATA_ROOT/pascal_voc" || exit 1

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf --strip-components=2 VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar