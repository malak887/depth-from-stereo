import argparse
import cv2 as cv2
import numpy as np
import read_calib
import calibrate as calibrate
import streaming
import computing_focal
import points_length


def parse_args():
    parser = argparse.ArgumentParser(
        description='ROV-26 Stereo Vision Pipeline — rectify, then measure 3D distance.'
    )
    parser.add_argument(
        '--left',
        default='frames/iceberg_prop1/camera0_31.png',
        help='Path to the left camera frame (default: frames/iceberg_prop1/camera0_31.png)',
    )
    parser.add_argument(
        '--right',
        default='frames/iceberg_prop1/camera1_31.png',
        help='Path to the right camera frame (default: frames/iceberg_prop1/camera1_31.png)',
    )
    parser.add_argument(
        '--out-dir',
        default='output',
        help='Directory for rectified images and other outputs (default: output)',
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run headless — skip all cv2.imshow() calls (useful for servers)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    

    cmtx0, cmtx1, dist0, dist1, R, T = read_calib.load_calibration()

    info = calibrate.rectify_and_save(
        cmtx0, dist0, cmtx1, dist1, R, T,
        left_img_path=args.left,
        right_img_path=args.right,
        out_dir=args.out_dir,
        show=not args.no_display,
    )

    result = points_length.measure_two_points(
        left_image_path=f"{args.out_dir}/rectified_left.png",
        right_image_path=f"{args.out_dir}/rectified_right.png",
    )

    print(f"\nPoint 1 pixel : {result['point1']}")
    print(f"Point 1 depth : {result['depth1']:.2f} cm")
    print(f"Point 1 3D    : {result['point1_3d']}")
    print(f"\nPoint 2 pixel : {result['point2']}")
    print(f"Point 2 depth : {result['depth2']:.2f} cm")
    print(f"Point 2 3D    : {result['point2_3d']}")
    print(f"\nDistance      : {result['distance']:.2f} cm")
