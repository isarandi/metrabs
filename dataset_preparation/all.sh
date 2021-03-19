#!/usr/bin/env bash
set -euo pipefail
source functions.sh

if [[ $(ask "The Human3.6M, MPII, Pascal VOC, MPI-INF-3DHP, MuCo-3DHP, MuPoTS and INRIA Holidays datasets are each from third parties.
 Have you read and agreed to their respective licenses? [y/n] ") != 'y' ]]; then
  echo "Then no cookies for you! Go read all the licenses!"
  exit 1
fi

cd "$(get_script_dir)"

./get_h36m.sh
./extract_frames_and_boxes_h36m.py

./get_mpii.sh
./setup_darknet.sh
darknet/run_yolo.sh --image-root "$DATA_ROOT/mpii" --out-path "$DATA_ROOT/mpii/yolov3_detections.pkl"

./get_pascal_voc.sh

./get_3dhp.sh
./extract_frames_and_masks_3dhp.py

./find_3dhp_images_for_detection.py > 3dhp_images_for_detection.txt
darknet/run_yolo.sh --image-paths-file 3dhp_images_for_detection.txt --out-path "$DATA_ROOT/3dhp/yolov3_person_detections.pkl"
rm 3dhp_images_for_detection.txt

./get_inria_holidays.sh
./prepare_images_inria_holidays.py
darknet/run_yolo.sh --image-root "$DATA_ROOT/inria_holidays/jpg_small" --out-path "$DATA_ROOT/inria_holidays.pkl"
./find_nonperson_images_inria_holidays.py

./get_muco.sh
darknet/run_yolo.sh --image-root "$DATA_ROOT/muco" --out-path "$DATA_ROOT/muco/yolov3_detections.pkl"
./postprocess_muco.py

./get_mupots.sh
./calibrate_mupots_intrinsics.py
darknet/run_yolo.sh --image-root "$DATA_ROOT/mupots" --out-path "$DATA_ROOT/mupots/yolov3_detections.pkl"

./get_3dpw.sh
darknet/run_yolo.sh --image-root "$DATA_ROOT/3dpw" --out-path "$DATA_ROOT/3dpw/yolov3_detections_noflip.pkl"
darknet/run_yolo.sh --image-root "$DATA_ROOT/3dpw" --out-path "$DATA_ROOT/3dpw/yolov3_detections_flip.pkl" --hflip
