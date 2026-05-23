import os
import yaml
import numpy as np
import cv2
from read_calib import read_intrinsics_file, read_rot_trans_file
DEFAULT_CAMERA_PARAMS_DIR = 'camera_parameters'
def load_intrinsics(path):
    """Load intrinsic matrix and distortion coefficients from YAML or legacy .dat."""
    if path.lower().endswith(('.yaml', '.yml')):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        cmtx = np.asarray(data['intrinsic'], dtype=np.float64)
        dist = np.asarray([data['distortion']], dtype=np.float64)
        return cmtx, dist
    return read_intrinsics_file(path)


def load_rot_trans(path):
    """Load rotation and translation from YAML or legacy .dat."""
    if path.lower().endswith(('.yaml', '.yml')):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        R = np.asarray(data['R'], dtype=np.float64)
        T = np.asarray(data['T'], dtype=np.float64).reshape((3, 1))
        return R, T
    return read_rot_trans_file(path)


def find_calibration_file(base_dir, prefix):
    """Find a YAML or .dat calibration file by prefix."""
    yaml_path = os.path.join(base_dir, prefix + '.yaml')
    yml_path = os.path.join(base_dir, prefix + '.yml')
    dat_path = os.path.join(base_dir, prefix + '.dat')
    for path in (yaml_path, yml_path, dat_path):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'Calibration file not found for {prefix} in {base_dir}')


def read_image(path):
    """Read an image using OpenCV if available, otherwise fall back to PIL."""
    if hasattr(cv2, 'imread'):
        return cv2.imread(path)
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError('Neither cv2.imread nor PIL.Image is available to load sample images.')

    img = Image.open(path)
    return np.array(img)


def infer_image_size():
    """Infer image size from the rectified stereo outputs."""
    left_path = 'output/rectified_left.png'
    right_path = 'output/rectified_right.png'

    if os.path.exists(left_path):
        img = read_image(left_path)
        if img is not None:
            return (img.shape[1], img.shape[0])

    if os.path.exists(right_path):
        img = read_image(right_path)
        if img is not None:
            return (img.shape[1], img.shape[0])

    raise ValueError(
        'Unable to infer image_size. Create rectified images in output/rectified_left.png or output/rectified_right.png.'
    )


class StereoCalibration:
    """Load stereo camera calibration parameters and compute rectification state."""

    def __init__(self, camera_parameters_dir=DEFAULT_CAMERA_PARAMS_DIR):
        self.camera_parameters_dir = camera_parameters_dir
        self.cmtx0 = None
        self.cmtx1 = None
        self.dist0 = None
        self.dist1 = None
        self.R = None
        self.T = None
        self.Q = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self._load_parameters()

    def _load_parameters(self):
        cam0_path = find_calibration_file(self.camera_parameters_dir, 'camera0_intrinsics')
        cam1_path = find_calibration_file(self.camera_parameters_dir, 'camera1_intrinsics')
        cam1_rt_path = find_calibration_file(self.camera_parameters_dir, 'camera1_rot_trans')

        self.cmtx0, self.dist0 = load_intrinsics(cam0_path)
        self.cmtx1, self.dist1 = load_intrinsics(cam1_path)
        self.R, self.T = load_rot_trans(cam1_rt_path)

    def compute_rectification(self, image_size, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY):
        if image_size is None:
            raise ValueError('image_size must be provided to compute rectification')

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.cmtx0,
            self.dist0,
            self.cmtx1,
            self.dist1,
            image_size,
            self.R,
            self.T,
            flags=flags,
            alpha=alpha,
        )

        self.Q = Q
        self.fx = P1[0, 0]
        self.fy = P1[1, 1]
        self.cx = P1[0, 2]
        self.cy = P1[1, 2]
        return R1, R2, P1, P2, Q

    def getQ(self, image_size=None, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY):
        if self.Q is None:
            if image_size is None:
                image_size = infer_image_size()
            self.compute_rectification(image_size, alpha=alpha, flags=flags)
        return self.fx, self.fy, self.cx, self.cy, self.Q


def getQ(camera_parameters_dir=DEFAULT_CAMERA_PARAMS_DIR, image_size=None, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY):
    """Return focal lengths, principal point, and Q matrix for the stereo setup."""
    calibrator = StereoCalibration(camera_parameters_dir)
    return calibrator.getQ(image_size=image_size, alpha=alpha, flags=flags)
