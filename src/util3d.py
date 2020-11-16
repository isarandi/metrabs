import logging
import queue
import sys

import matplotlib
import matplotlib.animation
import matplotlib.colors
import numpy as np
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

import data.h36m
import procrustes
import paths


def plot3d_worker(
        stick_figure_edges, has_ground_truth=True, interval=1000 / 50, batched=True, *, q):
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 6))
    fig.set_animated(True)
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.set_title('Input image')
    dummy_image = np.full([10, 10, 3], 0.5)
    image_artist = image_ax.imshow(dummy_image, animated=True, interpolation=None)

    sp1ax = fig.add_subplot(1, 2, 2, projection='3d')
    stick_plot_pred = StickPlot('Prediction', stick_figure_edges, sp1ax)
    fig.tight_layout()

    def iterate_frame_data():
        for data in poll_iterate_queue_till_none(q):
            if data is not None and batched:
                yield from zip(*data)
            else:
                yield data

        # We reach this point when the queue gives a None
        sys.exit(0)

    def animate(data):
        # None signals there is nothing new to plot
        if data is None:
            return

        im, *coord_arrays = data
        if not np.all(np.isnan(im)):
            image_artist.set_array(im)

        coord_arrays[0] -= np.nanmean(coord_arrays[0], axis=0, keepdims=True)
        if has_ground_truth:
            coord_arrays[1] -= np.nanmean(coord_arrays[1], axis=0, keepdims=True)
            stick_plot_pred.update(coord_arrays[0], ghost_coords=coord_arrays[1])
        else:
            stick_plot_pred.update(coord_arrays[0])

    animation = matplotlib.animation.FuncAnimation(
        fig, animate, frames=iterate_frame_data(), save_count=0, interval=interval, blit=False,
        repeat=False)
    plt.show(block=True)


class StickPlot:
    def __init__(self, title, stick_figure_edges, ax, elev=17, azim=47, rang=800):
        self.lines = []
        self.ghost_lines = []
        self.initialized = False
        self.ax = ax
        self.ax.set_title(title)
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlim3d(-rang, rang)
        self.ax.set_ylim3d(-rang, rang)
        self.ax.set_zlim3d(-rang, rang)
        self.ax.set_box_aspect((1, 1, 1))
        self.stick_figure_edges = stick_figure_edges

    def initialize(self, coords, ghost_coords=None):
        for i_start, i_end in self.stick_figure_edges:
            if ghost_coords is not None:
                line, = self.ax.plot(*zip(ghost_coords[i_start], ghost_coords[i_end]),
                                     color='grey', linestyle='--', marker='o', markersize=2)
                self.ghost_lines.append(line)

            line, = self.ax.plot(*zip(coords[i_start], coords[i_end]), marker='o', markersize=2)
            self.lines.append(line)

        self.initialized = True

    @staticmethod
    def _prepare_coords(coords):
        coords = np.vstack([coords, [0, 0, 0]])
        coords += [0, 0, 3000]
        # TODO this camera is not precisely the same as used when preparing the input (centering)
        cameras = data.h36m.get_cameras(f'{paths.DATA_ROOT}/h36m/Release-v1.2/metadata.xml')
        coords = cameras[0][0].camera_to_world(coords)
        coords -= coords[-1]
        return coords

    def update(self, coords, ghost_coords=None):
        coords = self._prepare_coords(coords)
        if ghost_coords is not None:
            ghost_coords = self._prepare_coords(ghost_coords)
        self.update_raw(coords, ghost_coords)

    def update_raw(self, coords, ghost_coords):
        if not self.initialized:
            return self.initialize(coords, ghost_coords)

        if ghost_coords is not None:
            for (i_start, i_end), line in zip(self.stick_figure_edges, self.ghost_lines):
                x, y, z = tuple(zip(ghost_coords[i_start], ghost_coords[i_end]))
                line.set_data(np.array(x), np.array(y))
                line.set_3d_properties(np.array(z))

        for (i_start, i_end), line in zip(self.stick_figure_edges, self.lines):
            x, y, z = tuple(zip(coords[i_start], coords[i_end]))
            line.set_data(np.array(x), np.array(y))
            line.set_3d_properties(np.array(z))


def poll_iterate_queue_till_none(q):
    """Returns a generator that iterates over a queue in a non-blocking (polling) way and yields
    None when the queue is empty. If the next element in the queue is None, the generator stops."""
    while True:
        try:
            data = q.get_nowait()
            if data is None:
                return
            yield data
        except queue.Empty:
            yield None


def rigid_align(coords_pred, coords_true, *, joint_validity_mask=None, scale_align=False,
                reflection_align=False):
    """Returns the predicted coordinates after rigid alignment to the ground truth."""

    if joint_validity_mask is None:
        joint_validity_mask = np.ones_like(coords_pred[..., 0], dtype=np.bool)

    valid_coords_pred = coords_pred[joint_validity_mask]
    valid_coords_true = coords_true[joint_validity_mask]
    try:
        d, Z, tform = procrustes.procrustes(
            valid_coords_true, valid_coords_pred, scaling=scale_align,
            reflection='best' if reflection_align else False)
    except np.linalg.LinAlgError:
        logging.error('Cannot do Procrustes alignment, returning original prediction.')
        return coords_pred

    T = tform['rotation']
    b = tform['scale']
    c = tform['translation']
    return b * coords_pred @ T + c


def rigid_align_many(
        coords_pred, coords_true, *, joint_validity_mask=None, scale_align=False,
        reflection_align=False):
    if joint_validity_mask is None:
        joint_validity_mask = np.ones_like(coords_pred[..., 0], dtype=np.bool)

    return np.stack([
        rigid_align(p, t, joint_validity_mask=jv, scale_align=scale_align,
                    reflection_align=reflection_align)
        for p, t, jv in zip(coords_pred, coords_true, joint_validity_mask)])
