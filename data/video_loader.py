"""
Video loading utility.

Reads a video file, samples a fixed number of frames evenly across its
length, and resizes them to a consistent shape for downstream models.
"""

import numpy as np
import cv2


def load_video_frames(video_path, num_frames=12, resize=(224, 224)):
    """
    Load a fixed number of frames from a video file.

    Args:
        video_path: path to the video file (mp4, webm, mkv, avi, mov)
        num_frames: number of frames to sample evenly across the video
        resize: (width, height) to resize each frame to

    Returns:
        np.ndarray of shape (num_frames, H, W, 3), dtype uint8, RGB order
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total <= 0:
        raise RuntimeError(f"Cannot read frames from {video_path}")

    idxs = np.linspace(0, total - 1, min(num_frames, total)).astype(int)

    frames, i = [], 0
    ret, frame = cap.read()

    while ret:
        if i in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
        ret, frame = cap.read()
        i += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

    # Pad by repeating the last frame if the video was shorter than requested
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.array(frames)
