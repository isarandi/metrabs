import itertools
import multiprocessing

import imageio
import more_itertools
import tensorflow as tf

import improc


def image_files_as_tf_dataset(
        image_paths, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False):
    width, height = improc.image_extents(image_paths[0])
    return image_dataset_from_queue(
        images_from_paths_gen, args=(image_paths,), imshape=[height, width], extra_data=extra_data,
        internal_queue_size=internal_queue_size, batch_size=batch_size, prefetch_gpu=prefetch_gpu,
        tee_cpu=tee_cpu)


def video_as_tf_dataset(
        video_path, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False):
    width, height = improc.video_extents(video_path)
    return image_dataset_from_queue(
        imageio.get_reader, args=(video_path,), imshape=[height, width], extra_data=extra_data,
        internal_queue_size=internal_queue_size, batch_size=batch_size, prefetch_gpu=prefetch_gpu,
        tee_cpu=tee_cpu)


def images_from_paths_gen(paths):
    yield from map(improc.imread_jpeg, paths)


def image_dataset_from_queue(
        generator_fn, imshape, extra_data, internal_queue_size, batch_size, prefetch_gpu, tee_cpu,
        args=None, kwargs=None):
    if internal_queue_size is None:
        internal_queue_size = batch_size if batch_size is not None else 64

    q = multiprocessing.Queue(internal_queue_size)
    t = multiprocessing.Process(
        target=queue_filler_process, args=(generator_fn, q, args, kwargs))
    t.start()

    def queue_reader():
        while (frame := q.get()) is not None:
            yield frame

    frames = queue_reader()
    if tee_cpu:
        frames, frames2 = itertools.tee(frames, 2)
    else:
        frames2 = itertools.repeat(None)

    ds = tf.data.Dataset.from_generator(lambda: frames, tf.uint8, [*imshape[:2], 3])

    if extra_data is not None:
        ds = tf.data.Dataset.zip((ds, extra_data))

    if batch_size is not None:
        ds = ds.batch(batch_size)
    if prefetch_gpu:
        ds = ds.apply(tf.data.experimental.prefetch_to_device('GPU:0', prefetch_gpu))

    if batch_size is not None:
        frames2 = more_itertools.chunked(frames2, batch_size)

    return ds, frames2


def queue_filler_process(generator_fn, q, args, kwargs):
    args = () if args is None else args
    kwargs = {} if kwargs is None else kwargs
    for item in generator_fn(*args, **kwargs):
        q.put(item)
    q.put(None)
