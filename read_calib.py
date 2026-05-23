import os
import numpy as np

def read_intrinsics_file(path):
    """Read camera matrix and distortion coefficients from a saved intrinsics file."""
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # Find sections
    i = lines.index('intrinsic:')
    j = lines.index('distortion:')
    
    # Parse camera matrix
    intrinsic_lines = lines[i+1:j]
    cmtx = np.array([[float(x) for x in row.split()] for row in intrinsic_lines])
    
    # Parse distortion coefficients
    distortion_line = lines[j+1]
    dist = np.array([[float(x) for x in distortion_line.split()]])
    
    return cmtx, dist

def read_rot_trans_file(path):
    """Read rotation matrix and translation vector from a saved rot_trans file."""
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # Find sections
    i = lines.index('R:')
    j = lines.index('T:')
    
    # Parse rotation matrix
    R_lines = lines[i+1:j]
    R = np.array([[float(x) for x in row.split()] for row in R_lines])
    
    # Parse translation vector
    T_lines = lines[j+1:]
    T_vals = []
    for row in T_lines:
        T_vals.extend([float(x) for x in row.split()])
    T = np.array(T_vals).reshape((3, 1))
    
    return R, T

def load_calibration():
    """Load all calibration parameters from camera_parameters folder."""
    cam0_intrinsics = os.path.join('camera_parameters', 'camera0_intrinsics.dat')
    cam1_intrinsics = os.path.join('camera_parameters', 'camera1_intrinsics.dat')
    cam1_rot_trans = os.path.join('camera_parameters', 'camera1_rot_trans.dat')
    
    # Read camera intrinsics
    cmtx0, dist0 = read_intrinsics_file(cam0_intrinsics)
    cmtx1, dist1 = read_intrinsics_file(cam1_intrinsics)
    
    # Read stereo parameters (R and T)
    R, T = read_rot_trans_file(cam1_rot_trans)
    
    return cmtx0, cmtx1, dist0, dist1, R, T

if __name__ == '__main__':
    # Example usage
    try:
        cmtx0, cmtx1, dist0, dist1, R, T = load_calibration()
        print("Successfully loaded calibration parameters:")
        print("\nCamera 0 matrix:")
        print(cmtx0)
        print("\nCamera 0 distortion:")
        print(dist0)
        print("\nCamera 1 matrix:")
        print(cmtx1)
        print("\nCamera 1 distortion:")
        print(dist1)
        print("\nRotation matrix:")
        print(R)
        print("\nTranslation vector:")
        print(T)
    except Exception as e:
        print(f"Error loading calibration parameters: {e}")