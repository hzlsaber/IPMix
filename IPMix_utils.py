"""Base augmentations operators."""

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
import random
import warnings
from torchvision import transforms

# suppress warnings
warnings.filterwarnings('ignore')

from torchvision import transforms

# ImageNet code should change this value to 224
IMAGE_SIZE = 32
# IMAGE_SIZE = 224

imageize = transforms.ToPILImage()
tensorize = transforms.ToTensor()


#########################################################
#################### AUGMENTATIONS ######################
#########################################################


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


#############Aug##############################

def invert(pil_img, _):
    return ImageOps.invert(pil_img)


def mirror(pil_img, _):
    return ImageOps.mirror(pil_img)

def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)

# operation that overlaps with ImageNet-C's test set
def autocontrast(pil_img, level):
    level = float_parameter(sample_level(level), 10)
    return ImageOps.autocontrast(pil_img, 10 - level)

# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

augmentations = [
   equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, mirror, invert
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness, mirror, invert
]


################################################################
######################## Pixels_MIXINGS ########################
################################################################

def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta, 1))
        b = np.float32(np.random.beta(1, beta))
    else:
        a = 1 + np.float32(np.random.beta(1, beta))
        b = -np.float32(np.random.beta(1, beta))
    return a, b


def add(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    out = (out + 1) / 2
    return out


def multiply(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1 ** a) * (img2.clip(1e-37) ** b)
    out = out / 2
    return out


def random_pixels(img1, img2, beta):
    (C, H, W) = img1.shape
    mask = torch.rand([1, H, W]) < torch.rand([])
    out = mask * img1 + (1 - mask.float()) * img2
    return out


def random_elems(img1, img2, beta):
    (C, H, W) = img1.shape
    mask = torch.rand([C, H, W]) < torch.rand([])
    out = mask * img1 + (1 - mask.float()) * img2
    return out

def random_mixing(img1, img2, beta):
    (C, H, W) = img1.shape
    if random.random()>0.5:
        mask = torch.rand([1, H, W]) < torch.rand([])
    else:
        mask = torch.rand([C, H, W]) < torch.rand([])
    out = mask * img1 + (1 - mask.float()) * img2
    return out

mixings = [add, multiply, random_pixels, random_elems]

################################################
############# Patch ############################
################################################
def patch_mixing(img, mixing_pic, patch_size, mixed_op, beta):
    org_w, org_h = img.size
    mask = None
    img_copy = img.copy()
    patch_left, patch_top = random.randint(*[0, org_w - patch_size]), random.randint(*[0, org_h - patch_size])
    if random.random() > 0.5:
        patch_right, patch_bottom = patch_left + patch_size, patch_top + patch_size
    else:
        patch_right = patch_left + random.randint(*[2, patch_size])
        patch_bottom = patch_top + random.randint(*[2, patch_size])
    img_patch = img.crop((patch_left, patch_top, patch_right, patch_bottom))
    mixing_patch = mixing_pic.crop((patch_left, patch_top, patch_right, patch_bottom))
    mixed_patch = imageize(mixed_op(tensorize(img_patch), tensorize(mixing_patch), beta))
    img_copy.paste(mixed_patch, (patch_left, patch_top), mask=mask)
    return img_copy


