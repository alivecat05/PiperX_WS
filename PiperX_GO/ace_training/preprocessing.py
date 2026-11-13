"""Checkpoint-compatible ACT input transforms shared by training and serving."""
import numpy as np

IMAGE_PREPROCESS = 'yuv420p_resize_rgb24_letterbox_v1'


def letterbox_rgb(frame, image_size):
    # Capture archives are yuv420p. Convert live RGB to that same source format
    # BEFORE resize; RGB-resize and YUV-resize aren't pixel equivalent.
    if frame.format.name != 'yuv420p':
        frame = frame.reformat(format='yuv420p')
    height, width = image_size
    ratio = min(width/frame.width, height/frame.height)
    w = min(width, max(1, round(frame.width*ratio)))
    h = min(height, max(1, round(frame.height*ratio)))
    rgb = frame.reformat(width=w, height=h, format='rgb24').to_ndarray()
    canvas = np.zeros((height,width,3), dtype=np.uint8)
    top,left = (height-h)//2,(width-w)//2
    canvas[top:top+h,left:left+w] = rgb
    return canvas


def normalize_state(state, stats):
    return (np.asarray(state,dtype=np.float32)-stats['qpos_mean'])/stats['qpos_std']


def image_to_float(images):
    import torch
    if images.dtype != torch.uint8:
        raise ValueError('ACT image input must be uint8 before its single /255 conversion')
    return images.float().div_(255.)
