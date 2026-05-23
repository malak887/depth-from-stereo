import cv2
import numpy as np
import yaml
from computing_focal import getQ


def _load_baseline_cm(settings_path='settings.yaml'):
    """Read baseline_cm from settings.yaml.

    Falls back to 12 cm with a warning if the key is missing, so older
    settings files don't cause a hard crash.
    """
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    if 'baseline_cm' not in settings:
        print(
            "WARNING: 'baseline_cm' not found in settings.yaml. "
            "Falling back to 12 cm. Add 'baseline_cm' to settings.yaml to remove this warning."
        )
        return 12.0
    return float(settings['baseline_cm'])


def _disparity_search_bounds(xL, img_width, window, search_range, T):
    """Return (start, end) for the horizontal SAD search in the right image.

    The search direction is determined by the sign of T[0] (the horizontal
    component of the translation between cameras):
      - T[0] < 0 means the right camera is to the right in world space →
        features appear at a SMALLER x in the right image → search LEFT.
      - T[0] > 0 (or unknown) → search RIGHT (original behaviour).

    Parameters
    ----------
    xL : int
        x-coordinate of the clicked point in the left image.
    img_width : int
        Width of the right image in pixels.
    window : int
        Half-size of the SAD patch window.
    search_range : int
        Number of pixels to search in each direction.
    T : numpy.ndarray
        Translation vector (3,1) from stereo calibration.
    """
   
    if T is not None and float(T[0]) < 0:
        # Standard horizontal stereo: right cam is physically to the right,
        # so disparity d = xL - xR > 0 → search leftward.
        start = max(window, xL - search_range)
        end = xL
    else:
        # Right cam is to the left, or direction unknown → search rightward.
        start = xL
        end = min(xL + search_range, img_width - window)
    return start, end


def match_patch(left_img, right_img, point, T=None, window=27, search_range=250):
    """Find the best horizontal match for a left-image patch in the right image.

    Parameters
    ----------
    left_img : numpy.ndarray
        Grayscale rectified left image.
    right_img : numpy.ndarray
        Grayscale rectified right image.
    point : tuple[int, int]
        (x, y) pixel in the left image to match.
    T : numpy.ndarray or None
        Translation vector from stereo calibration, used to determine the
        correct search direction. If None, defaults to rightward search.
    window : int
        Half-size of the SAD patch (patch is (2*window+1) × (2*window+1)).
    search_range : int
        Maximum number of pixels to search from the anchor point.

    Returns
    -------
    tuple[int, int] or None
        (xR, yL) of the best match, or None if the patch is out of bounds.
    """
    xL, yL = point

    patchL = left_img[yL - window:yL + window + 1, xL - window:xL + window + 1]
    if patchL.shape[0] == 0 or patchL.shape[1] == 0:
        return None

    best_score = 1e12
    best_xR = xL

    start, end = _disparity_search_bounds(xL, right_img.shape[1], window, search_range, T)

    for xR in range(start, end):
        patchR = right_img[yL - window:yL + window + 1, xR - window:xR + window + 1]
        if patchR.shape != patchL.shape:
            continue
        score = np.sum((patchL.astype('float32') - patchR.astype('float32')) ** 2)
        if score < best_score:
            best_score = score
            best_xR = xR

    return (best_xR, yL)


def compute_depth(clicked_point, settings_path='settings.yaml'):
    """Compute the metric depth of a clicked pixel using SAD patch matching.

    Parameters
    ----------
    clicked_point : tuple[int, int]
        (x, y) pixel coordinates in the rectified left image.
    settings_path : str
        Path to settings.yaml (must contain baseline_cm).

    Returns
    -------
    float or None
        Depth in centimetres, or None if matching failed or disparity is invalid.
    """
    path_L = "output/rectified_left.png"
    path_R = "output/rectified_right.png"

    imgL = cv2.imread(path_L)
    imgR = cv2.imread(path_R)

    if imgL is None:
        raise FileNotFoundError(f"Could not load left image: {path_L}")
    if imgR is None:
        raise FileNotFoundError(f"Could not load right image: {path_R}")

    left_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Load T so match_patch can pick the correct search direction
    try:
        from read_calib import read_rot_trans_file
        import os
        rt_path = os.path.join('camera_parameters', 'camera1_rot_trans.dat')
        _, T = read_rot_trans_file(rt_path)
    except Exception:
        T = None  # Fall back to rightward search if calibration unavailable

    matched = match_patch(left_gray, right_gray, clicked_point, T=T, window=13, search_range=250)

    if matched is None:
        print("No match found.")
        return None

    xR, y_matched = matched
    xL, yL = clicked_point

    # Show matched point on the right image for visual confirmation
    imgR_marked = imgR.copy()
    cv2.circle(imgR_marked, (xR, y_matched), 8, (0, 0, 255), 2)
    cv2.imshow("Matched Point (Right Image)", imgR_marked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # FIX: disparity sign must match search direction.
    # For leftward search (standard rig): d = xL - xR  (positive when xR < xL)
    # For rightward search (inverted rig): d = xR - xL (positive when xR > xL)
    if T is not None and float(T[0]) < 0:
        disp_manual = xL - xR
    else:
        disp_manual = xR - xL

    if disp_manual <= 0:
        print("Invalid disparity from matching (got %d). Check rig geometry or T[0] sign." % disp_manual)
        return None

    print("Disparity:", disp_manual)

    focal_px = getQ()[0]  # fx from rectified P1 matrix
    # FIX: baseline_cm is now read from settings.yaml instead of being hard-coded.
    baseline_cm = _load_baseline_cm(settings_path)

    depth_cm = (focal_px * baseline_cm) / disp_manual
    return depth_cm
