import logging

import cv2
import numpy as np
from typing import Tuple

LOSSLESS_CODECS = ['MPNG', 'FFV1', 'HFYU']
LOSSY_CODECS = ['DIVX', 'MJPG', 'XVID', 'mp4v', 'X264', 'x264', 'H264', 'h264']

class VideoStream:
    '''Class to write images to a video stream, with lossless or lossy compression.
    '''
    def __init__(self, fname: str, fps: int, lossless: bool):
        self.fps = fps
        self.fname = fname
        self.writer = None
        self.codecs = LOSSLESS_CODECS if lossless else LOSSY_CODECS

    def start_writer(self, size: Tuple[int, int]):
        writer = cv2.VideoWriter() 
        codec_found = False
        
        for codec in self.codecs:
            writer.open(self.fname, cv2.VideoWriter_fourcc(*codec), self.fps, size)
            if writer.isOpened():
                codec_found = True
                logging.info(f'Using codec {codec}')
                break
            else:
                logging.info(f'> Codec {codec} not supported, skipping.\n')
        if not codec_found:
            raise NotImplementedError('None of the supported codecs are available. Try installing ffmpeg.')

        self.writer = writer
        
    def write(self, image: np.ndarray):
        if self.writer is None:
            size = (image.shape[1], image.shape[0])
            self.start_writer(size)
        self.writer.write(np.uint8(image))

    def save(self):
        self.writer.release()
