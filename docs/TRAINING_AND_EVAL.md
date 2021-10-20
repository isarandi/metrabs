## Training

*The following applies as of the git commit 1aa1e756d*

Make sure to [install the dependencies](../install_dependencies.sh)
and [prepare the datasets](DATASETS.md)first (mind the licenses, some/most data is restricted to
academic research), then you can train the models. Here are a few example configurations
(append `--gui` to see visualizations):

### MeTRo

```bash
$ cd src
$ ./main.py \
    --train --dataset=h36m --train-on=trainval --epochs=27 --seed=1 \
    --logdir=h36m/metro_seed1

$ ./main.py \
    --train --dataset=h36m --train-on=trainval --epochs=27 --seed=1 \
    --scale-recovery=bone-lengths --logdir=h36m/2.5d_seed1

$ ./main.py \
    --train --dataset=mpi-inf-3dhp --train-on=trainval --epochs=27 --seed=1 \
    --background-aug-prob=0.7 --universal-skeleton --logdir=3dhp/metro_univ_seed1

$ ./main.py \
    --train --dataset=mpi-inf-3dhp --train-on=trainval --epochs=27 --seed=1 \
    --background-aug-prob=0.7 --no-universal-skeleton --logdir=3dhp/metro_nonuniv_seed1
```

### MeTRAbs

```bash
$ ./main.py \
    --train --dataset=muco-17-150k --dataset2d=mpii-yolo --scale-recovery=metrabs \
    --epochs=24 --seed=1 --background-aug-prob=0.7 --occlude-aug-prob=0.3 \ 
    --stride-test=32 --logdir=muco/metrabs_univ_seed1 --universal-skeleton
```

## Evaluation

To compute benchmark evaluation metrics, we first need to produce predictions on test data, then we
run the evaluation script on the prediction results. For example:

```bash
$ CHECKPOINT_DIR="$DATA_ROOT/experiments/h36m/someconfig"
$ ./main.py --test --dataset=h36m --stride-test=4 --checkpoint-dir="$CHECKPOINT_DIR"
$ python -m scripts.eval_h36m --pred-path="$CHECKPOINT_DIR/predictions_h36m.npz"
```

```bash
$ CHECKPOINT_DIR="$DATA_ROOT/experiments/3dhp/someconfig"
$ ./main.py --test --dataset=mpi-inf-3dhp --stride-test=4 --checkpoint-dir="$CHECKPOINT_DIR"
$ python -m scripts.eval_3dhp --pred-path="$CHECKPOINT_DIR/predictions_mpi-inf-3dhp.npz"
```

```bash
$ CHECKPOINT_DIR="$DATA_ROOT/experiments/muco/someconfig"
$ ./main.py --test --dataset=mupots --scale-recovery=metrabs --stride-test=32 --checkpoint-dir="$CHECKPOINT_DIR"
$ python -m scripts.eval_mupots --pred-path="$CHECKPOINT_DIR/predictions_mupots.npz"
```

The first command in each case creates the file `$CHECKPOINT_DIR/predictions_$DATASET.npz`.

Note: the script `eval_mupots.py` was designed and tested to produce the same results as Mehta et
al.'s official MuPoTS Matlab evaluation script. However, this Python version is much faster and
computes several different variations of the evaluation metrics at the same time
(only matched or all annotations, root relative or absolute, universal or metric-scale, bone
rescaling, number of joints).

To evaluate and average over multiple random seeds:

```bash
$ for s in {1..5}; do ./main.py --train --test --dataset=h36m --train-on=trainval --epochs=27 --seed=$i --logdir=h36m/metro_seed$i; done
$ python -m scripts.eval_h36m --pred-path="h36m/metro_seed1/predictions_h36m.npz" --seeds=5
```

## Packaging Models

We first export a single-person TensorFlow SavedModel that operates on batches of 256x256 px image
crops directly:

```bash
$ CHECKPOINT_DIR="$DATA_ROOT/experiments/muco/someconfig"
$ ./main.py --scale-recovery=metrabs --dataset=mupots --checkpoint-dir="$CHECKPOINT_DIR" --export-file="$CHECKPOINT_DIR"/metrabs_mupots_singleperson --data-format=NHWC --stride-train=32 --stride-test=32
```

Then we build a multiperson model which takes full images and multiple bounding boxes per image as
arguments:

```bash
$ python -m scripts.build_multiperson_model --input-model-path="$CHECKPOINT_DIR"/metrabs_mupots_singleperson --output-model-path="$CHECKPOINT_DIR"/metrabs_mupots_multiperson
```

We can also build a combined detector + pose estimator model as follows:

```bash
$ python -m scripts.build_combined_model --input-model-path="$CHECKPOINT_DIR"/metrabs_mupots_multiperson --detector-path=./yolov4 --output-model-path="$CHECKPOINT_DIR"/metrabs_mupots_multiperson_combined
```

## 3DPW Inference

To generate results on 3DPW and evaluate them, run

```bash
$ python -m scripts.video_inference --gt-assoc --dataset=3dpw --detector-path=./yolov4 --model-path=models/metrabs_multiperson_smpl --num-aug=5 --output-dir=./3dpw_predictions 
$ python -m scripts.eval_3dpw --pred-path=./3dpw_predictions
```