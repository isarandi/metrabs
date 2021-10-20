import copy
import functools
import warnings

import cv2
import numba
import numpy as np
import transforms3d

import cv2r


def point_transform(f):
    """Makes a function that transforms multiple points accept also a single point as well as
    lists, tuples etc. that can be converted by np.asarray."""

    def wrapped(self, points, *args, **kwargs):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            return np.squeeze(f(self, points[np.newaxis], *args, **kwargs), 0)
        else:
            return f(self, points, *args, **kwargs)

    return wrapped


def unit_vec(v):
    return v / np.linalg.norm(v)


class Camera:
    def __init__(
            self, optical_center=None, rot_world_to_cam=None, intrinsic_matrix=np.eye(3),
            distortion_coeffs=None, world_up=(0, 0, 1), extrinsic_matrix=None):
        """Pinhole camera with extrinsic and intrinsic calibration with optional distortions.

        The camera coordinate system has the following axes:
          x points to the right
          y points down
          z points forwards

        The world z direction is assumed to point up by default, but `world_up` can also be
         specified differently.

        Args:
            optical_center: position of the camera in world coordinates (eye point)
            rot_world_to_cam: 3x3 rotation matrix for transforming column vectors
                from being expressed in world reference frame to being expressed in camera
                reference frame as follows:
                column_point_cam = rot_matrix_world_to_cam @ (column_point_world - optical_center)
            intrinsic_matrix: 3x3 matrix that maps 3D points in camera space to homogeneous
                coordinates in image (pixel) space. Its last row must be (0,0,1).
            distortion_coeffs: parameters describing radial and tangential lens distortions,
                following OpenCV's model and order: k1, k2, p1, p2, k3 or None,
                if the camera has no distortion.
            world_up: a world vector that is designated as "pointing up".
            extrinsic_matrix: 4x4 extrinsic transformation matrix as an alternative to
                providing `optical_center` and `rot_world_to_cam`.
        """

        if optical_center is not None and extrinsic_matrix is not None:
            raise Exception('Cannot provide both `optical_center` and `extrinsic_matrix`!')
        if extrinsic_matrix is not None and rot_world_to_cam is not None:
            raise Exception('Cannot provide both `rot_world_to_cam` and `extrinsic_matrix`!')

        if (optical_center is None) and (extrinsic_matrix is None):
            optical_center = np.zeros(3, dtype=np.float32)

        if (rot_world_to_cam is None) and (extrinsic_matrix is None):
            rot_world_to_cam = np.eye(3, dtype=np.float32)

        if extrinsic_matrix is not None:
            self.R = np.asarray(extrinsic_matrix[:3, :3], dtype=np.float32)
            self.t = -self.R.T @ extrinsic_matrix[:3, 3].astype(np.float32)
        else:
            self.R = np.asarray(rot_world_to_cam, dtype=np.float32)
            self.t = np.asarray(optical_center, dtype=np.float32)

        self.intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float32)
        if distortion_coeffs is None:
            self.distortion_coeffs = None
        else:
            self.distortion_coeffs = np.asarray(distortion_coeffs, dtype=np.float32)

        self.world_up = np.asarray(world_up, dtype=np.float32)

        if not np.allclose(self.intrinsic_matrix[2, :], [0, 0, 1]):
            raise Exception(f'Bottom row of intrinsic matrix must be (0,0,1), '
                            f'got {self.intrinsic_matrix[2, :]}.')

    def get_distortion_coeffs(self):
        if self.distortion_coeffs is None:
            return np.zeros(shape=(5,), dtype=np.float32)
        return self.distortion_coeffs

    @staticmethod
    def create2D(imshape=(0, 0)):
        """Create a camera for expressing 2D transformations by using intrinsics only.

        Args:
            imshape: height and width, the principal point of the intrinsics is set at the middle
                of this image size.

        Returns:
            The new camera.
        """

        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[:2, 2] = [imshape[1] / 2, imshape[0] / 2]
        return Camera(intrinsic_matrix=intrinsics)

    def shift_image(self, offset):
        """Adjust intrinsics so that the projected image is shifted by `offset`.

        Args:
            offset: an (x, y) offset vector. Positive values mean that the resulting image will
                shift towards the left and down.
        """
        self.intrinsic_matrix[:2, 2] += offset

    def allclose(self, other_camera):
        """Check if all parameters of this camera are close to corresponding parameters
        of `other_camera`.

        Args:
            other_camera: the camera to compare to.

        Returns:
            True if all parameters are close, False otherwise.
        """
        return (np.allclose(self.intrinsic_matrix, other_camera.intrinsic_matrix) and
                np.allclose(self.R, other_camera.R) and np.allclose(self.t, other_camera.t) and
                allclose_or_nones(self.distortion_coeffs, other_camera.distortion_coeffs))

    def shift_to_desired(self, current_coords_of_the_point, target_coords_of_the_point):
        """Shift the principal point such that what's currently at `desired_center_image_point`
        will be shown at `target_coords_of_the_point`.

        Args:
            current_coords_of_the_point: current location of the point of interest in the image
            target_coords_of_the_point: desired location of the point of interest in the image
        """

        self.intrinsic_matrix[:2, 2] += (target_coords_of_the_point - current_coords_of_the_point)

    def reset_roll(self):
        """Roll the camera upright by turning along the optical axis to align the vertical image
        axis with the vertical world axis (world up vector), as much as possible.
        """

        self.R[:, 0] = unit_vec(np.cross(self.R[:, 2], self.world_up))
        self.R[:, 1] = np.cross(self.R[:, 0], self.R[:, 2])

    def orbit_around(self, world_point_pivot, angle_radians, axis='vertical'):
        """Rotate the camera around a vertical or horizontal axis passing through `world point` by
        `angle_radians`.

        Args:
            world_point_pivot: the world coordinates of the pivot point to turn around
            angle_radians: the amount to rotate
            axis: 'vertical' or 'horizontal'.
        """

        if axis == 'vertical':
            axis = self.world_up
        else:
            lookdir = self.R[2]
            axis = unit_vec(np.cross(lookdir, self.world_up))

        rot_matrix = cv2.Rodrigues(axis * angle_radians)[0]
        # The eye position rotates simply as any point
        self.t = (rot_matrix @ (self.t - world_point_pivot)) + world_point_pivot

        # R is rotated by a transform expressed in world coords, so it (its inverse since its a
        # coord transform matrix, not a point transform matrix) is applied on the right.
        # (inverse = transpose for rotation matrices, they are orthogonal)
        self.R = self.R @ rot_matrix.T

    def rotate(self, yaw=0, pitch=0, roll=0):
        """Rotate this camera by yaw, pitch, roll Euler angles in radians,
        relative to the current camera frame."""
        camera_rotation = transforms3d.euler.euler2mat(yaw, pitch, roll, 'ryxz')

        # The coordinates rotate according to the inverse of how the camera itself rotates
        point_coordinate_rotation = camera_rotation.T
        self.R = point_coordinate_rotation @ self.R

    @point_transform
    def camera_to_image(self, points):
        """Transform points from 3D camera coordinate space to image space.
        The steps involved are:
            1. Projection
            2. Distortion (radial and tangential)
            3. Applying focal length and principal point (intrinsic matrix)

        Equivalently:

        projected = points[:, :2] / points[:, 2:]

        if self.distortion_coeffs is not None:
            r2 = np.sum(projected[:, :2] ** 2, axis=1, keepdims=True)

            k = self.distortion_coeffs[[0, 1, 4]]
            radial = 1 + np.hstack([r2, r2 ** 2, r2 ** 3]) @ k

            p_flipped = self.distortion_coeffs[[3, 2]]
            tagential = projected @ (p_flipped * 2)
            distorted = projected * np.expand_dims(radial + tagential, -1) + p_flipped * r2
        else:
            distorted = projected

        return distorted @ self.intrinsic_matrix[:2, :2].T + self.intrinsic_matrix[:2, 2]
        """

        if self.distortion_coeffs is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', numba.NumbaPerformanceWarning)
                result = project_points(points, self.distortion_coeffs, self.intrinsic_matrix)
            return result
        else:
            projected = points[..., :2] / points[..., 2:]
            return projected @ self.intrinsic_matrix[:2, :2].T + self.intrinsic_matrix[:2, 2]

    @point_transform
    def world_to_camera(self, points):
        points = np.asarray(points, np.float32)
        return (points - self.t) @ self.R.T

    @point_transform
    def camera_to_world(self, points):
        points = np.asarray(points, np.float32)
        return points @ np.linalg.inv(self.R).T + self.t

    @point_transform
    def world_to_image(self, points):
        return self.camera_to_image(self.world_to_camera(points))

    @point_transform
    def image_to_camera(self, points, depth=1):
        if self.distortion_coeffs is None:
            normalized_points = (
                ((points - self.intrinsic_matrix[:2, 2]) @
                 np.linalg.inv(self.intrinsic_matrix[:2, :2])))
            return cv2.convertPointsToHomogeneous(normalized_points)[:, 0, :] * depth

        points = np.expand_dims(np.asarray(points, np.float32), 0)
        new_image_points = cv2.undistortPoints(
            points, self.intrinsic_matrix, self.distortion_coeffs, None, None, None)
        return cv2.convertPointsToHomogeneous(new_image_points)[:, 0, :] * depth

    @point_transform
    def image_to_world(self, points, camera_depth=1):
        return self.camera_to_world(self.image_to_camera(points, camera_depth))

    def is_visible(self, world_points, imsize):
        imsize = np.asarray(imsize)
        cam_points = self.world_to_camera(world_points)
        im_points = self.camera_to_image(cam_points)

        is_within_frame = np.all(np.logical_and(0 <= im_points, im_points < imsize), axis=1)
        is_in_front_of_camera = cam_points[..., 2] > 0
        return np.logical_and(is_within_frame, is_in_front_of_camera)

    def zoom(self, factor):
        """Zooms the camera (factor > 1 makes objects look larger),
        while keeping the principal point fixed (scaling anchor is the principal point)."""
        self.intrinsic_matrix[:2, :2] *= np.expand_dims(np.float32(factor), -1)

    def scale_output(self, factor):
        """Adjusts the camera such that the images become scaled by `factor`. It's a scaling with
        the origin as anchor point.
        The difference with `self.zoom` is that this method also moves the principal point,
        multiplying its coordinates by `factor`."""
        self.intrinsic_matrix[:2] *= np.expand_dims(np.float32(factor), -1)

    def undistort(self):
        self.distortion_coeffs = None

    def square_pixels(self):
        """Adjusts the intrinsic matrix such that the pixels correspond to squares on the
        image plane."""
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        fmean = 0.5 * (fx + fy)
        multiplier = np.array([[fmean / fx, 0, 0], [0, fmean / fy, 0], [0, 0, 1]], np.float32)
        self.intrinsic_matrix = multiplier @ self.intrinsic_matrix

    def horizontal_flip(self):
        self.R[0] *= -1

    def center_principal_point(self, imshape):
        """Adjusts the intrinsic matrix so that the principal point becomes located at the center
        of an image sized imshape (height, width)"""

        self.intrinsic_matrix[:2, 2] = np.float32([imshape[1] / 2, imshape[0] / 2])

    def shift_to_center(self, desired_center_image_point, imshape):
        """Shifts the principal point such that what's currently at `desired_center_image_point`
        will be shown in the image center of an image shaped `imshape`."""

        current_coords_of_the_point = desired_center_image_point
        target_coords_of_the_point = np.float32([imshape[1], imshape[0]]) / 2
        self.intrinsic_matrix[:2, 2] += (
                target_coords_of_the_point - current_coords_of_the_point)

    def turn_towards(self, target_image_point=None, target_world_point=None):
        """Turns the camera so that its optical axis goes through a desired target point.
        It resets any roll or horizontal flip applied previously. The resulting camera
        will not have horizontal flip and will be upright (0 roll)."""

        assert (target_image_point is None) != (target_world_point is None)
        if target_image_point is not None:
            target_world_point = self.image_to_world(target_image_point)

        new_z = unit_vec(target_world_point - self.t)
        new_x = unit_vec(np.cross(new_z, self.world_up))
        new_y = np.cross(new_z, new_x)

        # row_stack because we need the inverse transform (we make a matrix that transforms
        # points from one coord system to another), which is the same as the transpose
        # for rotation matrices.
        self.R = np.row_stack([new_x, new_y, new_z]).astype(np.float32)

    def get_projection_matrix(self):
        extrinsic_projection = np.append(self.R, -self.R @ np.expand_dims(self.t, 1), axis=1)
        return self.intrinsic_matrix @ extrinsic_projection

    def get_extrinsic_matrix(self):
        return np.block(
            [[self.R, -self.R @ np.expand_dims(self.t, -1)], [0, 0, 0, 1]]).astype(np.float32)

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def from_fov(fov_degrees, imshape):
        f = np.max(imshape[:2]) / (np.tan(np.deg2rad(fov_degrees) / 2) * 2)
        intrinsics = np.array(
            [[f, 0, imshape[1] / 2],
             [0, f, imshape[0] / 2],
             [0, 0, 1]], np.float32)
        return Camera(intrinsic_matrix=intrinsics)


def reproject_image_points(points, old_camera, new_camera):
    """Transforms keypoints of an image captured with `old_camera` to the corresponding
    keypoints of an image captured with `new_camera`.
    The world position (optical center) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output image."""

    if (old_camera.distortion_coeffs is None and new_camera.distortion_coeffs is None and
            points.ndim == 2):
        return reproject_image_points_fast(points, old_camera, new_camera)

    if not np.allclose(old_camera.t, new_camera.t):
        raise Exception(
            'The optical center of the camera must not change, else warping is not enough!')

    if (np.allclose(new_camera.R, old_camera.R) and
            allclose_or_nones(new_camera.distortion_coeffs, old_camera.distortion_coeffs)):
        relative_intrinsics = (
                new_camera.intrinsic_matrix @ np.linalg.inv(old_camera.intrinsic_matrix))
        return points @ relative_intrinsics[:2, :2].T + relative_intrinsics[:2, 2]

    world_points = old_camera.image_to_world(points)
    return new_camera.world_to_image(world_points)


def reproject_image(
        image, old_camera, new_camera, output_imshape, border_mode=cv2.BORDER_CONSTANT,
        border_value=0, interp=None, antialias_factor=1, dst=None):
    """Transform an `image` captured with `old_camera` to look like it was captured by
    `new_camera`. The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output.
    Ignores the issue of aliasing altogether.

    Args:
        image: the input image
        old_camera: the camera that captured `image`
        new_camera: the camera that should capture the newly returned image
        output_imshape: (height, width) for the output image
        border_mode: OpenCV border mode for treating pixels outside `image`
        border_value: OpenCV border value for treating pixels outside `image`
        interp: OpenCV interpolation to be used for resampling.
        antialias_factor: If larger than 1, first render a higher resolution output image
            that is `antialias_factor` times larger than `output_imshape` and subsequently resize
            it by 'area' interpolation to the desired size.
        dst: destination array (optional)

    Returns:
        The new image.
    """
    if antialias_factor == 1:
        return reproject_image_aliased(
            image, old_camera, new_camera, output_imshape, border_mode, border_value, interp,
            dst=dst)

    new_camera = new_camera.copy()
    a = antialias_factor
    new_camera.scale_output(a)
    new_camera.intrinsic_matrix[:2, 2] += (a - 1) / 2
    result = reproject_image_aliased(
        image, old_camera, new_camera, np.array(output_imshape) * a, border_mode, border_value,
        interp)

    return cv2r.resize(
        result, dsize=(output_imshape[1], output_imshape[0]),
        interpolation=cv2.INTER_AREA, dst=dst)


def reproject_image_aliased(
        image, old_camera, new_camera, output_imshape, border_mode=cv2.BORDER_CONSTANT,
        border_value=0, interp=None, dst=None):
    """Transform an `image` captured with `old_camera` to look like it was captured by
    `new_camera`. The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output.
    Aliasing issues are ignored.
    """

    if interp is None:
        interp = cv2.INTER_LINEAR

    if old_camera.distortion_coeffs is None and new_camera.distortion_coeffs is None:
        return reproject_image_fast(
            image, old_camera, new_camera, output_imshape, border_mode, border_value, interp, dst)

    if not np.allclose(old_camera.t, new_camera.t):
        raise Exception(
            'The optical center of the camera must not change, else warping is not enough!')

    output_size = (output_imshape[1], output_imshape[0])

    # 1. Simplest case: if only the intrinsics have changed we can use an affine warp
    if (np.allclose(new_camera.R, old_camera.R) and
            allclose_or_nones(new_camera.distortion_coeffs, old_camera.distortion_coeffs)):
        relative_intrinsics_inv = np.linalg.solve(
            new_camera.intrinsic_matrix.T, old_camera.intrinsic_matrix.T).T
        return cv2r.warpAffine(
            image, relative_intrinsics_inv[:2], output_size, flags=cv2.WARP_INVERSE_MAP | interp,
            borderMode=border_mode, borderValue=border_value, dst=dst)

    # 2. The general case handled by transforming the coordinates of every pixel
    # (i.e. computing the source pixel coordinates for each destination pixel)
    # and remapping (i.e. resampling the image at the resulting coordinates)
    new_maps = get_grid_coords((output_imshape[0], output_imshape[1]))
    newim_coords = new_maps.reshape([-1, 2])

    if new_camera.distortion_coeffs is None:
        partial_homography = (
                old_camera.R @ np.linalg.inv(new_camera.R) @
                np.linalg.inv(new_camera.intrinsic_matrix))
        new_im_homogeneous = cv2r.convertPointsToHomogeneous(newim_coords)
        old_camera_coords = new_im_homogeneous @ partial_homography.T
        oldim_coords = old_camera.camera_to_image(old_camera_coords)
    else:
        world_coords = new_camera.image_to_world(newim_coords)
        oldim_coords = old_camera.world_to_image(world_coords)

    old_maps = oldim_coords.reshape(new_maps.shape).astype(np.float32)
    map1 = old_maps[..., 0]
    map2 = old_maps[..., 1]

    if isinstance(image, cv2.cuda_GpuMat):
        if interp == cv2.INTER_NEAREST:
            # The Cuda version is off by a half pixel for nearest neightbor
            map1 += 0.5
            map2 += 0.5
        map1 = cv2.cuda_GpuMat(map1)
        map2 = cv2.cuda_GpuMat(map2)

    remapped = cv2r.remap(
        image, map1, map2, interp, borderMode=border_mode, borderValue=border_value, dst=dst)

    if not isinstance(image, cv2.cuda_GpuMat) and remapped.ndim < image.ndim:
        return np.expand_dims(remapped, -1)

    return remapped


def allclose_or_nones(a, b):
    """Check if all corresponding values in arrays a and b are close to each other in the sense of
    np.allclose, or both a and b are None, or one is None and the other is filled with zeros.
    """

    if a is None and b is None:
        return True

    if a is None:
        return np.min(b) == np.max(b) == 0

    if b is None:
        return np.min(a) == np.max(a) == 0

    return np.allclose(a, b)


@numba.jit(nopython=True)
def project_points(points, dist_coeff, intrinsic_matrix):
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    points = points.astype(np.float32)
    proj = points[..., :2] / points[..., 2:]
    r2 = np.sum(proj * proj, axis=1)
    distorter = (
            ((dist_coeff[4] * r2 + dist_coeff[1]) * r2 + dist_coeff[0]) * r2 +
            np.float32(1.0) + np.sum(proj * (np.float32(2.0) * dist_coeff[3:1:-1]), axis=1))
    proj[:] = (
            proj * np.expand_dims(distorter, 1) + np.expand_dims(r2, 1) * dist_coeff[3:1:-1])
    return (proj @ intrinsic_matrix[:2, :2].T + intrinsic_matrix[:2, 2]).astype(np.float32)


@functools.lru_cache()
def get_grid_coords(output_imshape):
    """Return a meshgrid of coordinates for the image shape`output_imshape` (height, width).

    Returns
        Meshgrid of shape [height, width, 2], with the x and y coordinates (in this order)
            along the last dimension. DType float32.
    """
    y, x = np.mgrid[:output_imshape[0], :output_imshape[1]].astype(np.float32)
    return np.stack([x, y], axis=-1)


def reproject_image_fast(
        image, old_camera, new_camera, output_imshape, border_mode=None, border_value=None,
        interp=cv2.INTER_LINEAR, dst=None):
    """Like reproject_image, but assumes there are no lens distortions."""

    old_matrix = old_camera.intrinsic_matrix @ old_camera.R
    new_matrix = new_camera.intrinsic_matrix @ new_camera.R
    homography = np.linalg.solve(new_matrix.T, old_matrix.T).T.astype(np.float32)

    if border_mode is None:
        border_mode = cv2.BORDER_CONSTANT
    if border_value is None:
        border_value = 0

    if isinstance(image, cv2.cuda_GpuMat):
        return cv2.cuda.warpPerspective(
            image, homography, (output_imshape[1], output_imshape[0]),
            flags=interp | cv2.WARP_INVERSE_MAP, borderMode=border_mode,
            borderValue=border_value)
    else:
        remapped = cv2.warpPerspective(
            image, homography, (output_imshape[1], output_imshape[0]),
            flags=interp | cv2.WARP_INVERSE_MAP, borderMode=border_mode,
            borderValue=border_value, dst=dst)

    if image.ndim == 2:
        return np.expand_dims(remapped, -1)
    return remapped


def reproject_image_points_fast(points, old_camera, new_camera):
    old_matrix = old_camera.intrinsic_matrix @ old_camera.R
    new_matrix = new_camera.intrinsic_matrix @ new_camera.R
    homography = np.linalg.solve(old_matrix.T, new_matrix.T).T.astype(np.float32)
    pointsT = homography[:, :2] @ points.T + homography[:, 2:]
    pointsT = pointsT[:2] / pointsT[2:]
    return pointsT.T


def get_affine(src_camera, dst_camera):
    """Return the affine transformation matrix that brings points from src_camera frame
    to dst_camera frame. Only works for in-plane rotations, translation and zoom.
    Throws if the transform would need a homography (due to out of plane rotation)."""

    # Check that the optical center and look direction stay the same
    if (not np.allclose(src_camera.t, dst_camera.t) or
            not np.allclose(src_camera.R[2], dst_camera.R[2])):
        raise Exception(
            'The optical center of the camera and its look '
            'direction may not change in the affine case!')

    src_points = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    dst_points = reproject_image_points(src_points, src_camera, dst_camera)
    return np.append(cv2.getAffineTransform(src_points, dst_points), [[0, 0, 1]], axis=0)
