# Inference

MeTRAbs, at its core, is based on single-person pose estimation and it is extended to multi-person
applications by first detecting people and then running the single-person estimation for each of
them. This so-called top-down multiperson strategy is already (efficiently) packaged into a
standalone model for convenience. This multiperson extension also takes into account the implicit
rotation that cropping induces. Instead of naive cropping, this model automatically takes care of
applying the appropriate homography transformation for perspective undistortion (optionally also
radial and tangential lens distortion), with anti-aliasing, and returns the poses in the world
coordinate frame (or camera frame if the extrinsic calibration is not given).

The models also contain built-in capability for test-time augmentation
(transforming each crop multiple times and averaging the results) and suppressing implausible
output.

In the following how-to, we show a few example calls. See also
thefull [API Reference](API.md).

## Using the built-in person detector

If you just have an image and know nothing else, you can use the ```detect_poses``` method which
uses the built-in YOLOv4 detector (based on https://github.com/hunglc007/tensorflow-yolov4-tflite)
to find person bounding boxes and estimate their poses, in a single method call.

```python
import tensorflow as tf

model = tf.saved_model.load('models/metrabs_eff2l_y4')
image = tf.image.decode_jpeg(tf.io.read_file('test_image_3dpw.jpg'))

# With unknown intrinsic matrix (assumes 55 degree field-of-view)
pred = model.detect_poses(image)
pred['boxes'], pred['poses2d'], pred['poses3d']

# Intrinsic matrix specified via field-of-view angle (of the larger image side).
# Principal point is assumed to be the image center.
pred = model.predict_single_image(image, default_fov_degrees=60)

# Known intrinsic matrix
intrinsic_matrix = tf.constant([[1962, 0, 540], [0, 1969, 960], [0, 0, 1]], dtype=tf.float32)
detections, poses3d, poses2d = model.detect_poses(image, intrinsic_matrix=intrinsic_matrix)
```

All the above also works for **multiple input images** (batched mode) using
the ```detect_poses_batched``` method. In this case the first argument needs to have
shape ```[batch_size, height, width, 3]```. For instance:

```python
# Just for demonstration we stack the same image twice. Here you would stack e.g. different
# frames of a video.
images = tf.stack([image, image], axis=0)

# One intrinsic matrix. This will be applied to all input images in the batch
intrinsic_matrix = tf.constant([[[1962, 0, 540], [0, 1969, 960], [0, 0, 1]]], dtype=tf.float32)
# The results are tf.RaggedTensors, since each input image may have a different number of 
# people in it.
pred = model.detect_poses_batched(image, intrinsic_matrix=intrinsic_matrix)
pred['boxes'], pred['poses2d'], pred['poses3d']

# Different intrinsic matrices for each input image in the batch:
intrinsic_matrix = tf.constant([
    [[1962, 0, 540], [0, 1969, 960], [0, 0, 1]],
    [[2521, 0, 530], [0, 2501, 970], [0, 0, 1]]], dtype=tf.float32)
pred = model.detect_poses_batched(image, intrinsic_matrix=intrinsic_matrix)
pred['boxes'], pred['poses2d'], pred['poses3d']
```

Since the user doesn't control the bounding boxes in this model, we can also perform a **
plausibility check** for each predicted pose and discard those that are not valid poses (based on
bone-lengths and augmentation consistency). Furthermore, duplicates are suppressed via **3D
pose-based non-maximum suppression**. (This allows setting the detector threshold lower to reduce
false negatives and at the same time eliminate most false positives.)

### Level 2: User-supplied bounding boxes

If you want to supply the bounding boxes yourself, you can do that as well with the
methods ```estimate_poses``` and ```estimate_poses_batched```:

```python
import tensorflow as tf

model = tf.saved_model.load('models/metrabs_eff2l_y4')
image = tf.image.decode_jpeg(tf.io.read_file('test_image_3dpw.jpg'))
intrinsic_matrix = tf.constant([[1962, 0, 540], [0, 1969, 960], [0, 0, 1]], dtype=tf.float32)
# Boxes are represented in [left, top, width, height] order
person_boxes = tf.constant(
    [[0, 626, 367, 896], [524, 707, 475, 841], [588, 512, 54, 198]], tf.float32)
pred = model.estimate_poses(image, boxes=person_boxes, intrinsic_matrix=intrinsic_matrix)
pred['poses2d'], pred['poses3d']
```

To predict **multiple images at once**, the bounding boxes have to be supplied in
a ```tf.RaggedTensor```, since each image in a batch may contain a different number of people.

```python
images = tf.stack([image, image], axis=0)
intrinsic_matrix = tf.constant([
    [[1962, 0, 540], [0, 1969, 960], [0, 0, 1]],
    [[2521, 0, 530], [0, 2501, 970], [0, 0, 1]]], dtype=tf.float32)

# Boxes are represented in [left, top, width, height] order
person_boxes = tf.ragged.constant([
    [[0, 626, 367, 896], [524, 707, 475, 841], [588, 512, 54, 198]],  # 3 boxes for the first image
    [[52, 514, 125, 741], [5, 160, 290, 414]],  # 2 boxes for the second image
], tf.float32, ragged_rank=1, inner_shape=(4,))

# The returned values are tf.RaggedTensors corresponding to the input box counts
pred = model.estimate_poses_batched(image, boxes=person_boxes, intrinsic_matrix=intrinsic_matrix)
pred['poses2d'], pred['poses3d']
```

### Bare-bones model, directly taking a batch of 256x256 px crops

If you want to perform the cropping yourself (it's tricky!), you can use the raw model as follows.
This model only works in batch mode and takes float16 images. This model is "bare-bones" in the
sense that a lot of the convenience features of the main model are not supported here
(e.g. skeleton convention selection, undistortion, etc.).

```python
import tensorflow as tf

model = tf.saved_model.load('models/metrabs_effnetv2l_raw')
image = tf.image.decode_jpeg(tf.io.read_file(...))  # a 256x256 px image
images = tf.stack([image, image], axis=0)
intrinsic_matrix = tf.constant(..., dtype=tf.float32)  # compute your own intrinsic matrix
poses3d = model.predict_multi(images, intrinsic_matrix)
```