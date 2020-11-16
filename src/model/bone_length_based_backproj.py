import numpy as np
import scipy.optimize
import tensorflow as tf

import tfu


def get_bone_lengths(coords3d, joint_info):
    return tf.stack([
        tf.norm(coords3d[:, i] - coords3d[:, j], axis=-1)
        for i, j in joint_info.stick_figure_edges], axis=1)


def optimize_z_offset_by_bones(xs, delta_zs, bone_lengths_ideal, edges):
    def fun(xs_, delta_zs_):
        return np.array([
            optimize_z_offset_by_bones_single(x, delta_z, bone_lengths_ideal, edges)
            for x, delta_z in zip(xs_, delta_zs_)], dtype=np.float32)

    batch_size = tfu.static_shape(xs)[0]
    return tfu.py_func_with_shapes(
        fun, [xs, delta_zs], output_types=(np.float32,), output_shapes=([batch_size],))[0]


def optimize_z_offset_by_bones_tensor(xs, delta_zs, bone_lengths_ideal, edges):
    def fun(xs_, delta_zs_, bone_lengths_ideal_):
        return np.array([
            optimize_z_offset_by_bones_single(x, delta_z, ll, edges)
            for x, delta_z, ll, w in zip(xs_, delta_zs_, bone_lengths_ideal_)],
            dtype=np.float32)

    batch_size = tfu.static_shape(xs)[0]
    return tfu.py_func_with_shapes(
        fun, [xs, delta_zs, bone_lengths_ideal], output_types=(np.float32,),
        output_shapes=([batch_size],))[0]


def optimize_z_offset_by_bones_single(
        x, delta_z, target_bone_lengths, edges, initial_guess=2000):
    """Given 2D points `x` in camera space (without intrinsics applied),
    depth coordinates up to shift (`delta_z`) and `target_bone_lengths` for the joint pairs
    `edges`, return the 3D skeleton constrained to have the given projection and delta_zs,
    while least-squares optimally matching the target bone lengths.
    """
    a = np.asarray([x[i] - x[j] for i, j in edges])
    y = x * np.expand_dims(delta_z, -1)
    b = np.asarray([y[i] - y[j] for i, j in edges])
    c = np.sum(a ** 2, axis=1)
    d = np.sum(2 * a * b, axis=1)
    e = np.sum(b ** 2, axis=1)

    def reconstruct_bone_lengths(z):
        return np.sqrt(z ** 2 * c + z * d + e)

    def fn(z):
        return (reconstruct_bone_lengths(z) - target_bone_lengths)

    def jacobian(z):
        return ((z * c + d) / reconstruct_bone_lengths(z)).reshape([-1, 1])

    solution = scipy.optimize.least_squares(fn, jac=jacobian, x0=initial_guess, method='lm')
    return float(solution.x)
