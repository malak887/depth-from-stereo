import cv2
import numpy as np
from computing_focal import getQ
from depth import compute_depth


def pixel_depth_to_3d(pt, z, fx, fy, cx, cy):
    """Convert a pixel coordinate and depth to 3D coordinates."""
    x, y = float(pt[0]), float(pt[1])
    if z is None or z == 0 or np.isnan(z):
        return None
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    return (X, Y, Z)


def distance_3d(a, b):
    """Compute Euclidean distance between two 3D points."""
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def select_point(image, window_name='Left Image'):
    """Display an image and return the pixel coordinates of a mouse click."""
    point = {'coords': None}

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point['coords'] = (x, y)
            print('Clicked:', point['coords'])

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1360, 720)
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, click_event)

    while point['coords'] is None:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
            break

    cv2.destroyWindow(window_name)
    return point['coords']


def measure_two_points(left_image_path='output/rectified_left.png', right_image_path='output/rectified_right.png'):
    """Measure the distance between two clicked points on a rectified left image."""
    fx, fy, cx, cy, Q = getQ()

    imgL = cv2.imread(left_image_path)
    if imgL is None:
        raise FileNotFoundError(f'Could not load left image: {left_image_path}')

    first_point = select_point(imgL, window_name='Select first point')
    if first_point is None:
        raise RuntimeError('First point selection cancelled.')
    depth1 = compute_depth(first_point)
    point1_3d = pixel_depth_to_3d(first_point, depth1, fx, fy, cx, cy)

    second_point = select_point(imgL, window_name='Select second point')
    if second_point is None:
        raise RuntimeError('Second point selection cancelled.')
    depth2 = compute_depth(second_point)
    point2_3d = pixel_depth_to_3d(second_point, depth2, fx, fy, cx, cy)

    distance = distance_3d(point1_3d, point2_3d)
    return {
        'point1': first_point,
        'depth1': depth1,
        'point1_3d': point1_3d,
        'point2': second_point,
        'depth2': depth2,
        'point2_3d': point2_3d,
        'distance': distance,
    }
