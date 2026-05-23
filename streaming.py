import cv2
import os
import yaml
import sys
from typing import Optional, Dict

with open('settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

# FIX: was settings.get('camera2') — key doesn't exist in settings.yaml,
# so DEFAULT_CAM_ID was always None, silently breaking live stream mode.
DEFAULT_CAM_ID = settings.get('camera0', 0)
DEFAULT_SAVE_DIR = "frames"
DEFAULT_FRAME_WIDTH = settings.get('frame_width')
DEFAULT_FRAME_HEIGHT = settings.get('frame_height')


def _next_index(save_dir: str, prefix: str) -> int:
    """Return the next available frame index for a given camera prefix."""
    if not os.path.exists(save_dir):
        return 0

    indices = []
    for f in os.listdir(save_dir):
        if f.startswith(prefix) and f.endswith('.png'):
            # FIX: bare `except:` swallowed all errors silently.
            # Only catch the specific parsing error we expect.
            try:
                num = int(f.split('_')[-1].split('.')[0])
                indices.append(num)
            except (ValueError, IndexError):
                continue

    return max(indices) + 1 if indices else 0


def stream(source_image_path: Optional[str] = None,
           cam_id: Optional[int] = None,
           save_dir: str = DEFAULT_SAVE_DIR,
           frame_width: Optional[int] = DEFAULT_FRAME_WIDTH,
           frame_height: Optional[int] = DEFAULT_FRAME_HEIGHT,
           interactive: bool = True) -> Dict[str, str]:
    """Capture stereo frames from a live camera or split a side-by-side image file.

    Parameters
    ----------
    source_image_path : str, optional
        Path to a side-by-side stereo image. If provided, skips live capture.
    cam_id : int, optional
        Camera device index for live capture. Defaults to camera0 from settings.yaml.
    save_dir : str
        Directory to save left/right frame pairs.
    frame_width : int, optional
        Capture width in pixels (combined side-by-side).
    frame_height : int, optional
        Capture height in pixels.
    interactive : bool
        If True, show a preview window during capture.

    Returns
    -------
    dict
        {'left': path_to_left_png, 'right': path_to_right_png}
    """
    os.makedirs(save_dir, exist_ok=True)
    last_saved = {'left': '', 'right': ''}

    def _save_pair(left_frame, right_frame, idx: int) -> Dict[str, str]:
        # If your rig requires a flip, set flip=True in the call site instead.
        left_name = os.path.join(save_dir, f"camera0_{idx}.png")
        right_name = os.path.join(save_dir, f"camera1_{idx}.png")

        cv2.imwrite(left_name, left_frame)
        cv2.imwrite(right_name, right_frame)

        print(f"Saved: {left_name}, {right_name}")
        return {'left': left_name, 'right': right_name}

    # ── Static image mode ──────────────────────────────────────────────────
    if source_image_path:
        frame = cv2.imread(source_image_path)
        if frame is None:
            raise IOError(f"Could not open image: {source_image_path}")

        h, w = frame.shape[:2]
        mid = w // 2
        left_frame = frame[:, :mid]
        right_frame = frame[:, mid:]

        idx = _next_index(save_dir, 'camera0_')
        saved = _save_pair(left_frame, right_frame, idx)
        last_saved.update(saved)

        if interactive:
            cv2.imshow("Stereo Image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return last_saved

    # ── Live stream mode ───────────────────────────────────────────────────
    cam_to_use = cam_id if cam_id is not None else DEFAULT_CAM_ID

    cam = cv2.VideoCapture(cam_to_use)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open stereo camera at device index {cam_to_use}.")

    if frame_width:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    if frame_height:
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    cv2.namedWindow("Stereo Stream", cv2.WINDOW_NORMAL)
    idx = _next_index(save_dir, 'camera0_')
    print("Streaming... Press 's' to save a frame pair, or ESC to exit.")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Frame capture failed.")
                break

            h, w = frame.shape[:2]
            mid = w // 2
            left_frame = frame[:, mid:]
            right_frame = frame[:, :mid]

            cv2.imshow("Stereo Stream", frame)
            key = cv2.waitKey(1)

            if key == ord('s'):
                saved = _save_pair(left_frame, right_frame, idx)
                last_saved.update(saved)
                idx += 1
            elif key == 27:  # ESC
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    print(f"Done. Saved images in '{save_dir}'.")
    return last_saved


if __name__ == '__main__':
    if len(sys.argv) > 1:
        stream(source_image_path=sys.argv[1], interactive=True)
    else:
        stream()
