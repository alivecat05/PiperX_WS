"""Image decoding shared by ACT and Diffusion Policy deployment."""
import av
import cv2
import numpy as np


class ImageInputs:
    def _image(self, data, allow_lossy=False, expected_shape=None, prepared=False):
        if isinstance(data, dict):
            if prepared:
                raise ValueError('Preprocessed payload must contain RGB pixels, not native YUV frames')
            if data.get('format') != 'yuv420p':
                raise ValueError('Unsupported native frame format')
            array = np.asarray(data['data'])
            if array.dtype != np.uint8 or array.ndim != 2 or array.shape[0]%3 or array.shape[1]%2:
                raise ValueError('Invalid native YUV420p frame')
            frame = av.VideoFrame.from_ndarray(array,format='yuv420p')
            if expected_shape and [frame.height,frame.width,3] != expected_shape:
                raise ValueError('Native camera frame size differs from training capture')
            return self.resize(frame,self.image_size)
        if isinstance(data, bytes):
            if data.startswith(b'\xff\xd8') and not allow_lossy:
                raise ValueError('JPEG changes model inputs; use lossless PNG/raw or explicitly allow lossy input')
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Invalid compressed image')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.asarray(data)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Images must be HWC uint8 RGB or JPEG/PNG bytes')
        if not 1 <= image.shape[0] <= 4096 or not 1 <= image.shape[1] <= 4096:
            raise ValueError('Image size is outside supported bounds')
        if expected_shape and list(image.shape) != expected_shape:
            raise ValueError(f'Camera frame size {list(image.shape)} differs from training {expected_shape}')
        if prepared:
            return image
        return self.resize(av.VideoFrame.from_ndarray(image, format='rgb24'), self.image_size)
