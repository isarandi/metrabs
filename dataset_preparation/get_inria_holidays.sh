#!/usr/bin/env bash
set -euo pipefail
source functions.sh
check_data_root

mkdir -p "$DATA_ROOT/inria_holidays"
cd "$DATA_ROOT/inria_holidays" || exit 1

for i in 1 2; do
  wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg$i.tar.gz
  tar -xvf jpg$i.tar.gz
  rm jpg$i.tar.gz
done
