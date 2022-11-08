import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

VISTA_DOWNSAMPLE_FACTOR = 4

from src.car_constants import IMAGE_CROP_XMIN, IMAGE_CROP_XMAX, IMAGE_CROP_YMIN, IMAGE_CROP_YMAX, \
                                    FULL_IMAGE_WIDTH, FULL_IMAGE_HEIGHT


def resize(cv_img, antialias):
    scale = 0.2
    height = IMAGE_CROP_YMAX - IMAGE_CROP_YMIN
    width = IMAGE_CROP_XMAX - IMAGE_CROP_XMIN

    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    img = torch.tensor(cv_img, dtype=torch.uint8).permute(2, 0, 1)
    img = F.resize(img, (scaled_height, scaled_width), antialias=antialias, interpolation=InterpolationMode.BILINEAR)
    return img.permute(1, 2, 0).numpy()

def normalise(img):
    return (img / 255.0)

def crop(cv_img, downsample_factor):
    crop_xmin = IMAGE_CROP_XMIN // downsample_factor
    crop_xmax = IMAGE_CROP_XMAX // downsample_factor
    crop_ymin = IMAGE_CROP_YMIN // downsample_factor
    crop_ymax = IMAGE_CROP_YMAX // downsample_factor
    return cv_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]

def crop_after_resize(cv_img, scale=0.2):
    crop_xmin = int(IMAGE_CROP_XMIN * scale)
    crop_xmax = int(IMAGE_CROP_XMAX * scale)
    crop_ymin = int(IMAGE_CROP_YMIN * scale)
    crop_ymax = int(IMAGE_CROP_YMAX * scale)

    return cv_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]

def preprocess(full_obs, antialias, resize_mode):

    if resize_mode == 'full':
        img = crop( full_obs, downsample_factor=1 )
        img = resize( img, antialias=antialias)
    elif resize_mode == 'downsample':
        img = crop( full_obs, downsample_factor=VISTA_DOWNSAMPLE_FACTOR )
        img = resize( img, antialias=antialias)
    elif resize_mode == 'resize':
        # full_obs already resized, only crop
        img = crop_after_resize( full_obs )
    elif resize_mode == 'resize_and_crop':
        # full_obs already resized and cropped
        img = full_obs

    img = normalise( img.astype(np.float32) )
    # Vista-synthesized images come from cv2, so need to convert BGR->RGB
    img = img[:, :, ::-1]
    return img

def grab_and_preprocess_obs(car, camera, antialias, resize_mode):
    full_obs = car.observations[camera.name]
    obs = preprocess(full_obs, antialias, resize_mode)
    return obs

def get_camera_size(resize_mode):
    if resize_mode == 'full':
        camera_size = (FULL_IMAGE_HEIGHT, FULL_IMAGE_WIDTH)
    elif resize_mode == 'downsample':
        camera_size = (FULL_IMAGE_HEIGHT // VISTA_DOWNSAMPLE_FACTOR, FULL_IMAGE_WIDTH // VISTA_DOWNSAMPLE_FACTOR)
    elif resize_mode == 'resize':
        camera_size = (FULL_IMAGE_HEIGHT // 5, FULL_IMAGE_WIDTH // 5)
    elif resize_mode == 'resize_and_crop':
        camera_size = (68, 264) # too hard to compute from the image size, just trust me

    return camera_size
