import cv2r
import imageio
import copy
import functools

import cv2
import numba
import numpy as np
import transforms3d

import boxlib


def support_single(f):
    """Makes a function that transforms multiple points accept also a single point"""

    def wrapped(self, points, *args, **kwargs):
        ndim = np.array(points).ndim
        if ndim == 1:
            return f(self, np.array([points]), *args, **kwargs)[0]
        else:
            return f(self, points, *args, **kwargs)

    return wrapped


class Camera:
    def __init__(
            self, optical_center=None, rot_world_to_cam=None, intrinsic_matrix=np.eye(3),
            distortion_coeffs=None, world_up=(0, 0, 1), extrinsic_matrix=None):
        """Initializes camera.

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
            world_up: a world vector that is designated as "pointing up", for use when
                the camera wants to roll itself upright.
        """

        if optical_center is not None and extrinsic_matrix is not None:
            raise Exception('At most one of `optical_center` and `extrinsic_matrix` needs to be '
                            'provided!')
        if extrinsic_matrix is not None and rot_world_to_cam is not None:
            raise Exception('At most one of `rot_world_to_cam` and `extrinsic_matrix` needs to be '
                            'provided!')

        if (optical_center is None) and (extrinsic_matrix is None):
            optical_center = np.zeros(3)

        if (rot_world_to_cam is None) and (extrinsic_matrix is None):
            rot_world_to_cam = np.eye(3)

        if extrinsic_matrix is not None:
            self.R = np.asarray(extrinsic_matrix[:3, :3], np.float32)
            self.t = (-self.R.T @ extrinsic_matrix[:3, 3]).astype(np.float32)
        else:
            self.R = np.asarray(rot_world_to_cam, np.float32)
            self.t = np.asarray(optical_center, np.float32)

        self.intrinsic_matrix = np.asarray(intrinsic_matrix, np.float32)
        if distortion_coeffs is None:
            self.distortion_coeffs = None
        else:
            self.distortion_coeffs = np.asarray(distortion_coeffs, np.float32)

        self.world_up = np.asarray(world_up)

        if not np.allclose(self.intrinsic_matrix[2, :], [0, 0, 1]):
            raise Exception(f'Bottom row of camera\'s intrinsic matrix must be (0,0,1), '
                            f'got {self.intrinsic_matrix[2, :]}.')

    @staticmethod
    def create2D(imshape=(0, 0)):
        intrinsics = np.eye(3)
        intrinsics[:2, 2] = [imshape[1] / 2, imshape[0] / 2]
        return Camera(intrinsic_matrix=intrinsics)

    def shift_image(self, offset):
        self.intrinsic_matrix[:2, 2] += offset

    def absolute_rotate(self, yaw=0, pitch=0, roll=0):
        def unit_vec(v):
            return v / np.linalg.norm(v)

        if self.world_up[0] > self.world_up[1]:
            world_forward = unit_vec(np.cross(self.world_up, [0, 1, 0]))
        else:
            world_forward = unit_vec(np.cross(self.world_up, [1, 0, 0]))
        world_right = np.cross(world_forward, self.world_up)

        R = np.row_stack([world_right, -self.world_up, world_forward]).astype(np.float32)
        mat = transforms3d.euler.euler2mat(-yaw, -pitch, -roll, 'syxz')
        self.R = mat @ R

    def allclose(self, other_camera):
        return (np.allclose(self.intrinsic_matrix, other_camera.intrinsic_matrix) and
                np.allclose(self.R, other_camera.R) and np.allclose(self.t, other_camera.t) and
                allclose_or_nones(self.distortion_coeffs, other_camera.distortion_coeffs))

    def shift_to_desired(self, current_coords_of_the_point, target_coords_of_the_point):
        """Shifts the principal point such that what's currently at `desired_center_image_point`
        will be shown in the image center of an image shaped `imshape`."""

        self.intrinsic_matrix[:2, 2] += (
                target_coords_of_the_point - current_coords_of_the_point)

    def reset_roll(self):
        def unit_vec(v):
            return v / np.linalg.norm(v)

        self.R[:, 0] = unit_vec(np.cross(self.R[:, 2], self.world_up))
        self.R[:, 1] = np.cross(self.R[:, 0], self.R[:, 2])

    def orbit_around(self, world_point, angle_radians, axis='vertical'):
        """Rotates the camera around a vertical axis passing through `world point` by
        `angle_radians`."""

        if axis == 'vertical':
            axis = self.world_up
        else:
            lookdir = self.R[2]
            axis = np.cross(lookdir, self.world_up)
            axis = axis / np.linalg.norm(axis)

        rot_matrix = cv2.Rodrigues(axis * angle_radians)[0]
        # The eye position rotates simply as any point
        self.t = (rot_matrix @ (self.t - world_point)) + world_point

        # R is rotated by a transform expressed in world coords, so it (its inverse since its a
        # coord transform matrix, not a point transform matrix) is applied on the right.
        # (inverse = transpose for rotation matrices, they are orthogonal)
        self.R = self.R @ rot_matrix.T

    def rotate(self, yaw=0, pitch=0, roll=0):
        mat = transforms3d.euler.euler2mat(yaw, pitch, roll, 'ryxz').T
        self.R = mat @ self.R

    @support_single
    def camera_to_image(self, points):
        """Transforms points from 3D camera coordinate space to image space.
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
            result = project_points(points, self.distortion_coeffs, self.intrinsic_matrix)
            return result
        else:
            projected = points[:, :2] / points[:, 2:]
            return projected @ self.intrinsic_matrix[:2, :2].T + self.intrinsic_matrix[:2, 2]

    @support_single
    def world_to_camera(self, points):
        points = np.asarray(points, np.float32)
        return (points - self.t) @ self.R.T

    @support_single
    def camera_to_world(self, points):
        points = np.asarray(points, np.float32)
        return points @ np.linalg.inv(self.R).T + self.t

    @support_single
    def world_to_image(self, points):
        return self.camera_to_image(self.world_to_camera(points))

    @support_single
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

    @support_single
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
        self.intrinsic_matrix[:2, :2] *= np.expand_dims(factor, -1)

    def scale_output(self, factor):
        """Adjusts the camera such that the images become scaled by `factor`. It's a scaling with
        the origin as anchor point.
        The difference with `self.zoom` is that this method also moves the principal point,
        multiplying its coordinates by `factor`."""
        self.intrinsic_matrix[:2] *= np.expand_dims(factor, -1)

    def undistort(self):
        self.distortion_coeffs = None

    def square_pixels(self):
        """Adjusts the intrinsic matrix such that the pixels correspond to squares on the
        image plane."""
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        fmean = 0.5 * (fx + fy)
        multiplier = np.array([[fmean / fx, 0, 0], [0, fmean / fy, 0], [0, 0, 1]])
        self.intrinsic_matrix = multiplier @ self.intrinsic_matrix

    def horizontal_flip(self):
        self.R[0] *= -1

    def center_principal_point(self, imshape):
        """Adjusts the intrinsic matrix so that the principal point becomes located at the center
        of an image sized imshape (height, width)"""

        self.intrinsic_matrix[:2, 2] = [imshape[1] / 2, imshape[0] / 2]

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

        def unit_vec(v):
            return v / np.linalg.norm(v)

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
        return np.block([[self.R, -self.R @ np.expand_dims(self.t, -1)], [0, 0, 0, 1]])

    def copy(self):
        return copy.deepcopy(self)


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
    """Transforms an image captured with `old_camera` to look like it was captured by
    `new_camera`. The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output."""

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
    if a is None and b is None:
        return True

    if a is None:
        return np.min(b) == np.max(b) == 0

    if b is None:
        return np.min(b) == np.max(b) == 0

    return np.allclose(a, b)


@numba.jit(nopython=True)
def project_points(points, dist_coeff, intrinsic_matrix):
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    points = points.astype(np.float32)
    proj = points[:, :2] / points[:, 2:]
    r2 = np.sum(proj * proj, axis=1)
    distorter = (
            ((dist_coeff[4] * r2 + dist_coeff[1]) * r2 + dist_coeff[0]) * r2 +
            np.float32(1.0) + np.sum(proj * (np.float32(2.0) * dist_coeff[3:1:-1]), axis=1))
    proj[:] = (
            proj * np.expand_dims(distorter, 1) + np.expand_dims(r2, 1) * dist_coeff[3:1:-1])
    return (proj @ intrinsic_matrix[:2, :2].T + intrinsic_matrix[:2, 2]).astype(np.float32)


@functools.lru_cache()
def get_grid_coords(output_imshape):
    y, x = np.mgrid[:output_imshape[0], :output_imshape[1]].astype(np.float32)
    return np.stack([x, y], axis=-1)


def reproject_image_fast(
        image, old_camera, new_camera, output_imshape, border_mode=None, border_value=None,
        interp=cv2.INTER_LINEAR, dst=None):
    """Like reproject_image, but assumes no distortions."""

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
