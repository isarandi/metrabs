"""Convenience wrappers over OpenCV functions."""

import os

import cv2
import numpy as np


def undistortPoints(src, cameraMatrix, distCoeffs, R=None, P=None):
    src = np.expand_dims(np.asarray(src, dtype=np.float32), 0)
    return cv2.undistortPoints(src, cameraMatrix, distCoeffs, None, R, P)[0]


def convertPointsToHomogeneous(src):
    return np.squeeze(cv2.convertPointsToHomogeneous(src), axis=1)


def convertPointsFromHomogeneous(src):
    return np.squeeze(cv2.convertPointsFromHomogeneous(src), axis=1)


def warpPerspective(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None):
    fun = cv2.cuda.warpPerspective if isinstance(src, cv2.cuda_GpuMat) else cv2.warpPerspective
    return fun(src, M, dsize, dst, flags, borderMode, borderValue)


def remap(src, map1, map2, interpolation, dst=None, borderMode=None, borderValue=None):
    fun = cv2.cuda.remap if isinstance(src, cv2.cuda_GpuMat) else cv2.remap
    return fun(src, map1, map2, interpolation, dst, borderMode, borderValue)


def warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None):
    fun = cv2.cuda.warpAffine if isinstance(src, cv2.cuda_GpuMat) else cv2.warpAffine
    return fun(src, M, dsize, dst, flags, borderMode, borderValue)


def resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None):
    fun = cv2.cuda.resize if isinstance(src, cv2.cuda_GpuMat) else cv2.resize
    return fun(src, dsize, dst, fx, fy, interpolation)
