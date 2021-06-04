# MeTRAbs Absolute 3D Human Pose Estimator

<a href="https://colab.research.google.com/github/isarandi/metrabs/blob/master/metrabs_demo.ipynb" target="_parent\"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a><br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/metrabs-metric-scale-truncation-robust/3d-human-pose-estimation-on-3d-poses-in-the)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3d-poses-in-the?p=metrabs-metric-scale-truncation-robust)

<p align="center"><img src=demo.png width="60%"></p>
<p align="center"><a href="https://youtu.be/BemM8-Lx47g"><img src=thumbnail_video.png width="30%"></a></p>

This repository contains code for the methods described in the following paper:

**[MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation](https://arxiv.org/abs/2007.07227)** <br>
*by István Sárándi, Timm Linder, Kai O. Arras, Bastian Leibe*<br>
IEEE Transactions on Biometrics, Behavior, and Identity Science (T-BIOM), Selected Best Works From Automatic Face and Gesture Recognition 2020 

## News
  * [2020-11-19] Oral presentation at the IEEE Conference on Automatic Face and Gesture Recognition (FG'20, online) ([Talk Video](https://youtu.be/BemM8-Lx47g) and [Slides](https://vision.rwth-aachen.de/media/papers/203/slides_metrabs.pdf))
  * [2020-11-16] Training and evaluation code now released along with dataset pre-processing scripts! Code and models upgraded to Tensorflow 2.
  * [2020-10-06] [Journal paper](https://arxiv.org/abs/2007.07227) accepted for publication in the IEEE Transactions on Biometrics, Behavior, and Identity Science (T-BIOM), Best of FG Special Issue
  * [2020-08-23] Short presentation at ECCV2020's 3DPW workshop ([slides](https://vision.rwth-aachen.de/media/papers/203/metrabs_3dpw_slides.pdf))
  * [2020-08-06] Our method has won the **[3DPW Challenge](https://virtualhumans.mpi-inf.mpg.de/3DPW_Challenge/)**
   
## Inference Code
We release **standalone TensorFlow models** (SavedModel) to allow easy application in downstream research. After loading the model, you can run inference in a single line of Python ([`demo.py`](demo.py)). You can try it in action in [Google Colab](https://colab.research.google.com/github/isarandi/metrabs/blob/master/metrabs_demo.ipynb).

(Note: These models were trained on the datasets cited in the paper. Since some are publicly available only for non-commercial use, the same restrictions may likely apply to the trained models as well. This is not legal advice.)

First, download and unzip the model files:

```bash
wget https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_{singleperson_smpl,multiperson_smpl,multiperson_smpl_combined}.zip -P ./models
unzip models/*.zip -d ./models
```
These models predict the 24 joints of the [SMPL body model](https://smpl.is.tue.mpg.de/). Stay tuned for further models (COCO, Human3.6M)!

If you **just have an image** and know nothing else, you can run the **combined detection+pose model** with a baked-in YOLOv4 detector (based on https://github.com/hunglc007/tensorflow-yolov4-tflite).
If the intrinsic matrix is not given, the angle of view is assumed to be 50 degrees (roughly appropriate for consumer cameras when zoomed out, but provide the true intrinsics for more accurate results.)

```python
import tensorflow as tf
model = tf.saved_model.load('./models/metrabs_multiperson_smpl_combined')
image = tf.image.decode_jpeg(tf.io.read_file('./test_image_3dpw.jpg'))
detections, poses3d, poses2d = model.predict_single_image(image)
```

If you want to **supply the bounding boxes yourself**, you can use the basic multiperson model:

```python
import tensorflow as tf
model = tf.saved_model.load('./models/metrabs_multiperson_smpl')
image = tf.image.decode_jpeg(tf.io.read_file('./test_image_3dpw.jpg'))
intrinsics = tf.constant([[1962, 0, 540], [0, 1969, 960], [0, 0, 1]], dtype=tf.float32)
# Boxes are represented as [x_left, y_top, width, height] arrays
person_boxes = tf.constant([[0, 626, 367, 896], [524, 707, 475, 841], [588, 512, 54, 198]], tf.float32)
poses3d = model.predict_single_image(image, intrinsics, person_boxes)
```

To perform inference on **multiple images at once**, use the `predict_multi_image` method. In this case, supply the bounding boxes
as a [tf.RaggedTensor](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor), since each image in the batch may contain a different number of detections. Intrinsics can be given as a 1x3x3 tensor, in which case the same matrix is used for all input images. Alternatively it can have shape `[n_images, 3, 3]` if each image has different intrinsics.

```python
import tensorflow as tf
model = tf.saved_model.load('./models/metrabs_multiperson_smpl')
images = tf.stack([
    tf.image.decode_jpeg(tf.io.read_file('./test_image1.jpg')),
    tf.image.decode_jpeg(tf.io.read_file('./test_image2.jpg'))], axis=0)

intrinsics = tf.constant([[[1962, 0, 540], [0, 1969, 960], [0, 0, 1]]], dtype=tf.float32)
person_boxes = tf.ragged.constant([[[0, 626, 367, 896], [524, 707, 475, 841]], [[588, 512, 54, 198]]], tf.float32)
poses3d = model.predict_multi_image(images, intrinsics, person_boxes)
```

MeTRAbs, at its core, is based on single-person pose estimation and it is extended to multi-person applications by 
first detecting people and then running the single-person estimation for each of them. This so-called **top-down multiperson 
strategy** is already (quite efficiently) packaged into a standalone model for convenience. This multiperson extension
also takes into account the implicit rotation that cropping induces. Instead of naive cropping, the model takes care
of applying the appropriate **homography transformation for perspective undistortion** and returns the poses in the 3D camera coordinate frame corresponding to the full image. The models also contain built-in capability for test-time augmentation
(transforming each crop `n` times and averaging the results, this is configurable through the argument `n_aug` to the prediction functions).

The individual, per-person crops will be **automatically batched up behind the scenes** and sent to the 
single-person pose estimator in that batched form for efficient parallelization on the GPU.
One such concrete batch may contain crops from multiple images (when using `predict_multi_image`), 
but the crops from one image may also be split in several batches if there are many boxes given. You can control this using the `internal_batch_size` argument to the above functions (0 means all crops are packed into one batch, which will lead to out-of-memory with many boxes).


## Training

To train models, make sure to [install the dependencies](install_dependencies.sh) and [prepare the datasets](DATASETS.md)
 first (mind the licenses, some/most data is restricted to academic research). Here are a few example configurations
 (append `--gui` to see visualizations during training):

### MeTRo (root-relative model)

```bash
cd src
./main.py \
  --train --dataset=h36m --train-on=trainval --epochs=27 --seed=1 \
  --logdir=h36m/metro_seed1

./main.py \
  --train --dataset=h36m --train-on=trainval --epochs=27 --seed=1 \
  --scale-recovery=bone-lengths --logdir=h36m/2.5d_seed1

./main.py \
  --train --dataset=mpi-inf-3dhp --train-on=trainval --epochs=27 --seed=1 \
  --background-aug-prob=0.7 --universal-skeleton --logdir=3dhp/metro_univ_seed1

./main.py \
  --train --dataset=mpi-inf-3dhp --train-on=trainval --epochs=27 --seed=1 \
  --background-aug-prob=0.7 --no-universal-skeleton --logdir=3dhp/metro_nonuniv_seed1
```

### MeTRAbs (absolute model including root estimation)

```bash
./main.py \
  --train --dataset=muco-17-150k --dataset2d=mpii-yolo --scale-recovery=metrabs \
  --epochs=24 --seed=1 --background-aug-prob=0.7 --occlude-aug-prob=0.3 \ 
  --stride-test=32 --logdir=muco/metrabs_univ_seed1 --universal-skeleton
```

## Evaluation

To compute benchmark evaluation metrics, we first need to produce predictions on test data,
 then we run the corresponding evaluation script on the generated prediction results.
 For example:

```bash
CHECKPOINT_DIR="$DATA_ROOT/experiments/h36m/someconfig"
./main.py --test --dataset=h36m --stride-test=4 --checkpoint-dir="$CHECKPOINT_DIR"
python -m scripts.eval_h36m --pred-path="$CHECKPOINT_DIR/predictions_h36m.npz"
```

```bash
CHECKPOINT_DIR="$DATA_ROOT/experiments/3dhp/someconfig"
./main.py --test --dataset=mpi-inf-3dhp --stride-test=4 --checkpoint-dir="$CHECKPOINT_DIR"
python -m scripts.eval_3dhp --pred-path="$CHECKPOINT_DIR/predictions_mpi-inf-3dhp.npz"
```

```bash
CHECKPOINT_DIR="$DATA_ROOT/experiments/muco/someconfig"
./main.py --test --dataset=mupots --scale-recovery=metrabs --stride-test=32 --checkpoint-dir="$CHECKPOINT_DIR"
python -m scripts.eval_mupots --pred-path="$CHECKPOINT_DIR/predictions_mupots.npz"
```

The first command in each case creates the file `$CHECKPOINT_DIR/predictions_$DATASET.npz`.

Note: the script [`eval_mupots.py`](src/scripts/eval_mupots.py) was designed and tested to produce the same results as Mehta et al.'s official [MuPoTS Matlab evaluation script](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/). 
 However, this Python version is much faster and computes several different variations of the evaluation metrics at the same time 
 (only matched vs. all annotations, root relative vs. absolute, universal vs. metric-scale, bone rescaling, number of joints).

To evaluate and average over multiple random seeds, as done in the paper:

```bash
for s in {1..5}; do ./main.py --train --test --dataset=h36m --train-on=trainval --epochs=27 --seed=$i --logdir=h36m/metro_seed$i; done
python -m scripts.eval_h36m --pred-path="h36m/metro_seed1/predictions_h36m.npz" --seeds=5
```

## Packaging Models

To make it easy to reuse these models in downstream research, we can make use of TensorFlow's SavedModels, which capture the entire model (graph) and the trained weights, so that the original Python code is no longer needed to run inference.

We first export a single-person TensorFlow SavedModel that operates on batches of 256x256 px image crops directly:

```bash
CHECKPOINT_DIR="$DATA_ROOT/experiments/muco/someconfig"
./main.py --scale-recovery=metrabs --dataset=mupots --checkpoint-dir="$CHECKPOINT_DIR" --export-file="$CHECKPOINT_DIR"/metrabs_mupots_singleperson --data-format=NHWC --stride-train=32 --stride-test=32
```

Then we build a multiperson model, which takes full (uncropped) images and multiple bounding boxes per image as its input arguments and outputs poses: 

```bash
python -m scripts.build_multiperson_model --input-model-path="$CHECKPOINT_DIR"/metrabs_mupots_singleperson --output-model-path="$CHECKPOINT_DIR"/metrabs_mupots_multiperson
```

We can also build a combined detector + pose estimator model as follows:

```bash
python -m scripts.build_combined_model --input-model-path="$CHECKPOINT_DIR"/metrabs_mupots_multiperson --detector-path=./yolov4 --output-model-path="$CHECKPOINT_DIR"/metrabs_mupots_multiperson_combined
```

## 3DPW Inference

To generate results on 3DPW and to evaluate them, run

```bash
python -m scripts.video_inference --gt-assoc --dataset=3dpw --detector-path=./yolov4 --model-path=models/metrabs_multiperson_smpl --crops=5 --output-dir=./3dpw_predictions 
python -m scripts.eval_3dpw --pred-path=./3dpw_predictions
```

The above [`eval_3dpw`](src/scripts/eval_3dpw.py) script is equivalent to the [official script](https://github.com/aymenmir1/3dpw-eval/) but it is much more efficient (however it does not evaluate joint angles, just joint positions).

## Cite as

```bibtex
@article{Sarandi20TBIOM,
  title={{MeTRAbs:} Metric-Scale Truncation-Robust Heatmaps for Absolute 3{D} Human Pose Estimation},
  author={S\'ar\'andi, Istv\'an and Linder, Timm and Arras, Kai O. and Leibe, Bastian},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2020}
  note={in press}
}
```

The above paper is an extended journal version of the FG'2020 conference paper:

```bibtex
@inproceedings{Sarandi20FG,
  title={Metric-Scale Truncation-Robust Heatmaps for 3{D} Human Pose Estimation},
  author={S\'ar\'andi, Istv\'an and Linder, Timm and Arras, Kai O. and Leibe, Bastian},
  booktitle={IEEE International Conference on Automatic Face and Gesture Recognition},
  pages={677-684},
  year={2020}
}
```

## Contact

Code in this repository was written by [István Sárándi](https://isarandi.github.io) (RWTH Aachen University) unless indicated otherwise.

Got any questions or feedback? Drop a mail to sarandi@vision.rwth-aachen.de!
