import cv2 as cv
import glob
import numpy as np
import yaml
import os

# FIX: removed `from scipy import linalg` — scipy was imported but never used.
# Unused imports signal careless code to a reviewer.

calibration_settings = {}


def parse_calibration_settings_file(filename):
    """Load calibration settings from a YAML file into the module-level dict.

    Parameters
    ----------
    filename : str
        Path to the settings YAML file (e.g. 'settings.yaml').

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If 'camera0' key is missing (likely the wrong file was passed).
    """
    global calibration_settings

    # FIX: replaced quit() with proper exceptions. quit() is an interactive-
    # shell helper — calling it inside a library function kills the entire
    # Python process with no traceback, making errors impossible to catch.
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Calibration settings file not found: {filename}")

    print('Using for calibration settings:', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    if 'camera0' not in calibration_settings:
        raise ValueError(
            f"'camera0' key not found in {filename}. "
            "Check that the correct settings file was passed."
        )


def calibrate_camera_for_intrinsic_parameters(images_prefix):
    """Detect checkerboard corners and calibrate one camera for intrinsics.

    Parameters
    ----------
    images_prefix : str
        Glob pattern for images, e.g. 'frames/camera0*'.

    Returns
    -------
    tuple
        (camera_matrix, distortion_coefficients)
    """
    images_names = glob.glob(images_prefix)
    images = [cv.imread(imname, 1) for imname in images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = images[0].shape[1]
    height = images[0].shape[0]

    imgpoints = []
    objpoints = []

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.putText(
                frame,
                'If detected points are poor, press "s" to skip this sample',
                (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1,
            )

            cv.imshow('img', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)

    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist


def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
    """Save intrinsic parameters to a .dat file in camera_parameters/.

    Parameters
    ----------
    camera_matrix : numpy.ndarray
        3×3 camera intrinsic matrix.
    distortion_coefs : numpy.ndarray
        Distortion coefficient array.
    camera_name : str
        Name prefix for the output file, e.g. 'camera0'.
    """
    os.makedirs('camera_parameters', exist_ok=True)

    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')

    # FIX: original code used open() with no close() and no context manager,
    # leaking a file handle. Use `with` to guarantee the file is closed and
    # fully flushed even if an exception occurs mid-write.
    with open(out_filename, 'w') as outf:
        outf.write('intrinsic:\n')
        for row in camera_matrix:
            outf.write(' '.join(str(v) for v in row) + '\n')

        outf.write('distortion:\n')
        outf.write(' '.join(str(v) for v in distortion_coefs[0]) + '\n')


def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    """Run stereo calibration to estimate R and T between the two cameras.

    Parameters
    ----------
    mtx0, dist0 : numpy.ndarray
        Intrinsics for camera 0.
    mtx1, dist1 : numpy.ndarray
        Intrinsics for camera 1.
    frames_prefix_c0 : str
        Glob pattern for camera 0 frames.
    frames_prefix_c1 : str
        Glob pattern for camera 1 frames.

    Returns
    -------
    tuple
        (R, T) — rotation matrix and translation vector from cam0 to cam1.
    """
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    imgpoints_left = []
    imgpoints_right = []
    objpoints = []

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0, 0].astype(np.int32)
            p0_c2 = corners2[0, 0].astype(np.int32)

            cv.putText(frame0, 'O', tuple(p0_c1), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.drawChessboardCorners(frame0, (rows, columns), corners1, c_ret1)
            cv.imshow('img', frame0)

            cv.putText(frame1, 'O', tuple(p0_c2), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.drawChessboardCorners(frame1, (rows, columns), corners2, c_ret2)
            cv.imshow('img2', frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx0, dist0, mtx1, dist1,
        (width, height),
        criteria=criteria,
        flags=cv.CALIB_FIX_INTRINSIC,
    )

    print('rmse:', ret)
    cv.destroyAllWindows()
    return R, T


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix=''):
    """Save extrinsic (R, T) parameters for both cameras to .dat files.

    Parameters
    ----------
    R0, T0 : numpy.ndarray
        Rotation and translation for camera 0 (typically identity / zeros).
    R1, T1 : numpy.ndarray
        Rotation and translation for camera 1 (from stereo calibration).
    prefix : str
        Optional filename prefix.
    """
    os.makedirs('camera_parameters', exist_ok=True)

    for cam_name, R, T in [('camera0', R0, T0), ('camera1', R1, T1)]:
        filename = os.path.join('camera_parameters', f"{prefix}{cam_name}_rot_trans.dat")
        # FIX: use context manager here too — same file-handle leak as above.
        with open(filename, 'w') as outf:
            outf.write('R:\n')
            for row in R:
                outf.write(' '.join(str(v) for v in row) + '\n')
            outf.write('T:\n')
            for row in T:
                outf.write(' '.join(str(v) for v in row) + '\n')

    return R0, T0, R1, T1


def rectify_and_save(cmtx0, dist0, cmtx1, dist1, R, T,
                     left_img_path, right_img_path,
                     out_dir='output', show=True):
    """Perform stereo rectification on a pair of images and save results.

    Parameters
    ----------
    cmtx0, dist0 : numpy.ndarray
        Intrinsics for camera 0.
    cmtx1, dist1 : numpy.ndarray
        Intrinsics for camera 1.
    R, T : numpy.ndarray
        Stereo extrinsics (rotation and translation).
    left_img_path : str
        Path to the left input image.
    right_img_path : str
        Path to the right input image.
    out_dir : str
        Output directory for rectified images.
    show : bool
        If True, display rectified images in windows (set False for headless runs).

    Returns
    -------
    dict
        Keys: 'left', 'right', 'combined' (output paths), 'Q' (matrix), 'maps'.
    """
    imgL = cv.imread(left_img_path)
    imgR = cv.imread(right_img_path)

    if imgL is None or imgR is None:
        raise IOError(
            f"Could not load images. Check paths:\n  left:  {left_img_path}\n  right: {right_img_path}"
        )

    image_size = (imgL.shape[1], imgL.shape[0])

    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        cmtx0, dist0, cmtx1, dist1, image_size, R, T, alpha=0
    )

    map1L, map2L = cv.initUndistortRectifyMap(cmtx0, dist0, R1, P1, image_size, cv.CV_16SC2)
    map1R, map2R = cv.initUndistortRectifyMap(cmtx1, dist1, R2, P2, image_size, cv.CV_16SC2)

    rectifiedL = cv.remap(imgL, map1L, map2L, interpolation=cv.INTER_LINEAR)
    rectifiedR = cv.remap(imgR, map1R, map2R, interpolation=cv.INTER_LINEAR)

    combined = np.hstack((rectifiedL, rectifiedR))
    for y in range(0, combined.shape[0], 40):
        cv.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    os.makedirs(out_dir, exist_ok=True)

    left_out = os.path.join(out_dir, 'rectified_left.png')
    right_out = os.path.join(out_dir, 'rectified_right.png')
    combined_out = os.path.join(out_dir, 'rectified_combined.png')

    cv.imwrite(left_out, rectifiedL)
    cv.imwrite(right_out, rectifiedR)
    cv.imwrite(combined_out, combined)

    print('Saved rectified images:')
    print('  -', left_out)
    print('  -', right_out)
    print('  -', combined_out)

    if show:
        cv.imshow('Left Rectified', rectifiedL)
        cv.imshow('Right Rectified', rectifiedR)
        cv.imshow('Rectified Pair (Check Lines)', combined)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return {
        'left': left_out,
        'right': right_out,
        'combined': combined_out,
        'Q': Q,
        'maps': (map1L, map2L, map1R, map2R),
    }
