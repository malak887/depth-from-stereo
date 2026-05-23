import cv2
import numpy as np


def get_disparity(imgL, imgR):
    """Compute a WLS-filtered disparity map from a rectified stereo pair.

    Parameters
    ----------
    imgL : str
        Path to the rectified left image.
    imgR : str
        Path to the rectified right image.

    Returns
    -------
    numpy.ndarray
        Normalised uint8 disparity map, ready for display or further processing.
    """
    left = cv2.imread(imgL)
    right = cv2.imread(imgR)

    if left is None:
        raise FileNotFoundError(f"Could not load left image: {imgL}")
    if right is None:
        raise FileNotFoundError(f"Could not load right image: {imgR}")

    left = cv2.equalizeHist(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    right = cv2.equalizeHist(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))

    # Slightly blur to reduce noise
    left = cv2.GaussianBlur(left, (5, 5), 0)
    right = cv2.GaussianBlur(right, (5, 5), 0)

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16,
        blockSize=9,
        P1=8 * 6 * 7,
        P2=32 * 8 * 7,
        disp12MaxDiff=20,
        uniquenessRatio=30,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lmbda = 90000.0
    sigma = 1.8

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = np.int16(left_matcher.compute(left, right).astype(np.float32))
    dispr = np.int16(right_matcher.compute(right, left).astype(np.float32))

    filteredImg = wls_filter.filter(displ, left, None, dispr)
    filteredImg = cv2.normalize(
        src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX
    )
    return np.uint8(filteredImg)


# FIX: module-level get_disparity() call and cv2.imshow() were executing on
# every import, opening GUI windows as a side-effect. Moved inside __main__.
if __name__ == '__main__':
    filtered = get_disparity("output/rectified_left.png", "output/rectified_right.png")
    cv2.imshow("Disparity Map", filtered)
    cv2.imshow("Disparity (colour)", cv2.applyColorMap(filtered, cv2.COLORMAP_JET))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
