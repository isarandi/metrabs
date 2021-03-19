**Note: You don't need to perform all these steps if you simply want to do inference. See the 'Inference' section in the main README.md.**

## Dataset Preparation
To run the experiments (or similar ones) as described in the paper, you need to download and prepare the datasets. This howto will guide you through all of that.

Completing *all* the downloads and *all* the preprocessing described here will take **up to a week or more** on a fast computer.
If you just want to train on Human3.6M it will take substantially less time.
It can be parallelized across machines, but this is not included here for simplicity.

The intended audience of this guide should already be familiar with Bash, Python and GNU/Linux command line
 tools, because it will likely need some tweaking for your particular environment.
While these scripts try to be helpful, they aren't bulletproof production-grade software. Don't blame me if it formats your hard drives / blows up your computer / eats your lunch / etc.!
 
**Disclaimer:** These are third party datasets, so make sure you understand the licenses involved 
and cite the appropriate papers when using them. 
I don't host the datasets, I merely wrote these scripts to help with downloading them. 
Contact the creators directly for questions about the datasets themselves.

### All at once
There is a script here called  `all.sh`, which contains **all the steps**.
In an ideal world the following would be all you need to run: 

```bash
$ export DATA_ROOT=/some/path1
$ export CUDA_ROOT=/some/path2/cuda-10.0
$ export CUDNN_ROOT=/some/path3/cudnn-7.6
$ git clone https://github.com/isarandi/metrabs.git
$ bash metrabs/dataset_preparation/all.sh
```

Still, read the whole guide because the process is fairly complex and will likely need some babysitting to go smoothly.
  
----

### Before starting

* Set the environment variable **`$DATA_ROOT`** to the path of a directory, where you have **at least around 100 GB free disk space** (TODO: find out the exact space needed).
* `cd` into the datasets directory: 
```bash
$ export DATA_ROOT=/path/to/enough/space
$ cd somepath/metrabs/dataset_preparation
```

The code will create subdirectories under `$DATA_ROOT` so make sure it's writable.

I've tested it on Ubuntu 18.04.5, Nvidia driver 450, CUDA 10.0, cuDNN 7.6.1 with an RTX 2080 Ti GPU. 


### Human3.6M
* Sign up on the official site: http://vision.imar.ro/human3.6m
* Get the dataset with my script:

```bash
$ ./get_h36m.sh
$ ./extract_frames_and_bounding_boxes_h36m.py
```

* Some more preprocessing will be done automatically in the main code on first run (cropping and resizing the relevant image parts for faster loading during training).

### PASCAL-VOC
* For use in synthetic occlusion augmentation. Official site: http://host.robots.ox.ac.uk/pascal/VOC/
* Get it with the script:

```bash
$ ./get_pascal_voc.sh
```

### MPII
* This is a 2D dataset, used for weak supervision
* Official site: http://human-pose.mpi-inf.mpg.de/#download

```bash
$ ./get_mpii.sh
```

### MPI-INF-3DHP
* Official site: http://gvv.mpi-inf.mpg.de/3dhp-dataset/

```bash 
$ ./get_3dhp.sh
$ ./extract_frames_and_masks_3dhp.py
```

* Boxes are not given in this dataset, so we create them with the YOLOv3 detector. (I patched it to remove image creation and to print out the box coordinates.)

```bash
$ ./setup_darknet.sh
$ ./find_3dhp_images_for_detection.py > 3dhp_images_for_detection.txt
$ darknet/run_yolo.sh --image-paths-file 3dhp_images_for_detection.txt --out-path "$DATA_ROOT/3dhp/yolov3_person_detections.pkl"
```

* About 2% (5314/235484) of the images will not have any detections. In most cases this is fine, there really is nobody in some of the images.
* As with H36M, some more preprocessing will be done automatically in the main code on first run (cropping and resizing the relevant image parts for faster loading during training).


### INRIA Holidays
* Official site: http://lear.inrialpes.fr/people/jegou/data.php
* These are used for background augmentation.

```bash
$ ./get_inria_holidays.sh
$ ./prepare_images_inria_holidays.py
```

* We don't want to confuse our network with people in the background, so we exclude person images based on YOLOv3's output.
```bash
$ darknet/run_yolo.sh --image-root "$DATA_ROOT/inria_holidays/jpg_small" --out-path "$DATA_ROOT/inria_holidays.pkl" --jobs 3 --hflip
$ ./find_nonperson_images_inria_holidays.py
```

### MuCo-3DHP
* This dataset is generated from MPI-INF-3DHP using a Matlab script released by its authors.
* Official site: http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson
* Generating it takes very long.

```bash
$ ./get_muco.sh
$ PYTHONPATH=../src ./postprocess_muco.py
```

### MuPoTS-3D
* Official site: http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson

```bash
$ ./get_mupots.sh
```

### 3DPW

* Official site: https://virtualhumans.mpi-inf.mpg.de/3DPW

```bash
$ ./get_3dpw.sh
```
