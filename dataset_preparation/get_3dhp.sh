#!/usr/bin/env bash
set -euo pipefail
source functions.sh
check_data_root

cd "$DATA_ROOT"
wget http://gvv.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip
unzip mpi_inf_3dhp.zip
rm mpi_inf_3dhp.zip
mv mpi_inf_3dhp 3dhp
cd 3dhp

sed -i 's/subjects=(1 2)/subjects=(1 2 3 4 5 6 7 8)/' conf.ig
sed -i "s/destination='.\/'/'destination=$DATA_ROOT\/3dhp\/'/" conf.ig
sed -i "s/ready_to_download=0/ready_to_download=1/" conf.ig
bash get_dataset.sh

bash get_testset.sh
mv mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/TS* ./
mv mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/test_util ./
mv mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/README.txt ./README_testset.txt
rmdir mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set
rmdir mpi_inf_3dhp_test_set
