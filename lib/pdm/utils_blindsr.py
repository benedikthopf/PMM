# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch

from lib.pdm import utils_image as util

import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
from PIL import Image, ImageEnhance
import logging
import string


"""
This code is adapted from https://github.com/cszn/BSRGAN
"""

class PDM_DatasetWrapper:
    def __init__(self, dataset, config, split = "train"):
        self.dataset = dataset
        assert split in {"train", "val", "test"}, "split must be 'train' 'val' or 'test"
        self.p = config["degradations_p"] if split == "train" else config["val_degradations_p"]
        self.strength = config["degradations_strength"]
        self.use_beta = config["use_beta"] if "use_beta" in config else False
        if self.use_beta:
            self.a = config["a"]
            self.b = config["b"]

        self.seen_shapes = set()

        self.degradation_type = config["type"]
        self.degradation_fns = {
            "bsrgan": lambda x: degradation_bsrgan_plus(x, sf=2, strength=self.get_strength())[0], 
            # "double": lambda x: degradation(x, strength=self.get_strength(), p = self.p), 
            # "single": lambda x: degradation(x, strength=self.get_strength(), p = self.p, single=True), 
            "ddrc": lambda x: degradation_ddrc(x),
            "ours": lambda x: self.degradation(x, strength=self.get_strength(), p=self.p, distractor_p=config["distractor_p"]),
            "noise": lambda x: self.add_fixed_level_noise(x, level=self.get_strength()),
            "blur": lambda x: self.add_fixed_level_blur(x, level=self.get_strength()),
            "jpeg": lambda x: self.add_fixed_level_jpeg(x, level=self.get_strength()),
            "resize": lambda x: self.add_fixed_level_resize(x, level=self.get_strength()),
            "motion": lambda x: self.add_fixed_level_motion(x, level=self.get_strength()),
            "brightness": lambda x: self.add_fixed_level_brightness(x, level=self.get_strength()),
            "speckle": lambda x: self.add_fixed_level_speckle(x, level=self.get_strength()),
        }

        self.degradation_fn = self.degradation_fns[self.degradation_type]

    def get_strength(self):
        if self.use_beta:
            return np.random.beta(self.a, self.b)
        else:
            return self.strength
    

    def __getitem__(self, i):
        x = self.dataset[i]
        if np.random.rand() < self.p or self.degradation_type in {"ours", "double", "single"}:
            x = self.degrade(x)
        return x


    def degrade(self, x):
        if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
            return self.transpose_degrade_transpose(x)
        elif isinstance(x, list) or isinstance(x, tuple):
            return [self.degrade(z) for z in x]
        elif isinstance(x, dict):
            return {k: self.degrade(v) for k, v in x.items()}
        elif isinstance(x, int) or x is None or isinstance(x, str):
            return x
        else:
            raise NotImplementedError(f"Cannot degrade object of type {type(x)}")

    def transpose_degrade_transpose(self, x):
        have_to_retransform = False
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if hasattr(x, "shape") and len(x.shape) == 3 and x.shape[0] == 3:
            if x.max() > 1 or x.min() < 0:
                x = x/2 + 0.5
                have_to_retransform = True
                if x.max() > 1 or x.min() < 0:
                    raise RuntimeError(f"Wrong image range [{x.min()}, {x.max()}]")
            x = torch.tensor(np.transpose(
                    self.degradation_fn(np.transpose(x, (1, 2, 0))),
                    (2, 0, 1)))
            if have_to_retransform:
                x = x * 2 - 1
            return x
        else:
            if hasattr(x, "shape"):
                if x.shape not in self.seen_shapes:
                    logging.info(f"Not degrading object with shape {x.shape}")
                    self.seen_shapes.add(x.shape)
            else:
                logging.info(f"Not degrading object of type {type(x)}")
            return x

    def __getattr__(self, i):
        return getattr(self.dataset, i)
    
    def __len__(self):
        return len(self.dataset)
    
    def add_fixed_level_noise(self, img, level):
        return (img + np.random.normal(0, level/255.0, img.shape)).clip(0,1)

    def add_fixed_level_jpeg(self, img, level):
        quality_factor = level
        img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
        result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        img = cv2.imdecode(encimg, 1)
        img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)
        return img
        
    def add_fixed_level_blur(self, img, level):
        k = anisotropic_Gaussian(ksize=level*2+1, theta=0, l1=level, l2=level)

        img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

        return img
    
    def add_fixed_level_resize(self, img, level):
        w, h, _ = img.shape
        img = cv2.resize(img, (int(level*w), int(level*h)), interpolation=1)
        img = cv2.resize(img, (int(w), int(h)), interpolation=1)
        
        return img
    
    def add_fixed_level_brightness(self, img, level):
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        enhance = ImageEnhance.Brightness(img)

        img = enhance.enhance(level)
        img = np.array(img).astype(float) / 255
        return img

    def add_fixed_level_speckle(self, img, level):
        noise_level = level
        img = np.clip(img, 0.0, 1.0)
        img += img*np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
        return img
    
    def add_fixed_level_motion(self, img, level):
        k = np.ones((1, level)) / level

        img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

        return img

    
    def degradation(self, img, strength=0.5, p=0.5, distractor_p=0):
        h, w, _ = img.shape

        shuffle_order = random.sample(range(8), 8)

        for i in shuffle_order:

            if i == 0:
                # blur
                if random.random() < p:
                    img = add_smoothing2(img, strength=strength)

            elif i == 1:
                # fancy noises
                if random.random() < p:
                    if random.random() < 0.5:
                        img = add_speckle_noise(img, strength=strength)
                    else:
                        img = add_Poisson_noise(img)
                
            elif i == 2:
                # resize
                if random.random() < p:
                    img = add_resize(img, sf=2, strength=strength)

            elif i == 3:
                # add strong Gaussian noise
                if random.random() < p/2:
                    img = add_Gaussian_noise(img, noise_level1=80, noise_level2=100, strength=strength)

            elif i == 4:
                # add Gaussian noise
                if random.random() < p/2:
                    img = add_Gaussian_noise(img, noise_level1=2, noise_level2=100, strength=strength)

            elif i == 5:
                # add JPEG noise
                if random.random() < p:
                    img = add_JPEG_noise(img)

            elif i == 6:
                # add enahnce
                if random.random() < p:
                    img = add_enhance(img)

            elif i == 7:
                count = 0
                while random.random() < distractor_p:
                    if count > 10:
                        break
                    count += 1

                    if random.random() < 0.5:
                        color = (
                            random.random(),
                            random.random(),
                            random.random(),
                        )
                        text_len = random.randint(0, 10)
                        text = "".join(np.random.choice(list(string.printable), text_len, replace=True))

                        img = cv2.putText(
                            img = np.ascontiguousarray(img),
                            text = text, 
                            org = (random.randint(-100, w), random.randint(0, h+100)), 
                            fontFace = np.random.randint(8), 
                            fontScale = random.random() * 8, 
                            color = color, 
                            thickness = random.randint(1, 8),
                            lineType = np.random.randint(3)
                        )
                    else:
                        h_now, w_now, _ = img.shape
                        try:
                            i2 = random.randint(0, len(self.dataset)-1)
                            img2, *_ = self.dataset[i2]
                            img2 = img2.permute(1, 2, 0).detach().cpu().numpy()
                            if img2.max() > 1 or img2.min() < 0:
                                img2 = img2/2 + 0.5
                        except Exception as e:
                            raise e
                        img2 = np.ascontiguousarray(img2)
                        small_size_x = random.randint(20, 100)
                        small_size_y = int(small_size_x*(random.random()*0.4 + 0.8))
                        img2 = cv2.resize(img2, (small_size_x, small_size_y), interpolation=random.choice([1, 2, 3]))

                        x, y = random.randint(-small_size_x, w_now), random.randint(-small_size_y, h_now)
                        
                        if x < 0:
                            img2 = img2[:, -x:, :]
                            small_size_x = small_size_x+x
                            x = 0
                        elif x+small_size_x >= w_now:
                            img2 = img2[:, :w_now-x, :]
                            small_size_x = w_now-x

                        if y < 0:
                            img2 = img2[-y:, :, :]
                            small_size_y = small_size_y+y
                            y = 0
                        elif y+small_size_y >= h_now:
                            img2 = img2[:h_now-y, :, :]
                            small_size_y = h_now-y

                        try:
                            img[y:y+small_size_y, x:x+small_size_x, :] = img2
                        except Exception as e:
                            raise RuntimeError("Size of small image does not match size of cutout for it to be placed in " + str((x, y, small_size_x, small_size_y, img2.shape, img[y:y+small_size_y, x:x+small_size_x, :].shape, img.shape)))



        # img = add_JPEG_noise(img, quality_factor=70)

        img = cv2.resize(img, (h, w), interpolation=random.choice([1, 2, 3]))

        return img




"""
# --------------------------------------------
# Super-Resolution
# --------------------------------------------
#
# Kai Zhang (cskaizhang@gmail.com)
# https://github.com/cszn
# From 2019/03--2021/08
# --------------------------------------------
"""

def modcrop_np(img, sf):
    '''
    Args:
        img: numpy image, WxH or WxHxC
        sf: scale factor

    Return:
        cropped image
    '''
    w, h = img.shape[:2]
    im = np.copy(img)
    return im[:w - w % sf, :h - h % sf, ...]


"""
# --------------------------------------------
# anisotropic Gaussian kernels
# --------------------------------------------
"""
def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.

    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def blur(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, Nx1xhxw
    '''
    n, c = x.shape[:2]
    p1, p2 = (k.shape[-2]-1)//2, (k.shape[-1]-1)//2
    x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode='replicate')
    k = k.repeat(1,c,1,1)
    k = k.view(-1, 1, k.shape[2], k.shape[3])
    x = x.view(1, -1, x.shape[2], x.shape[3])
    x = torch.nn.functional.conv2d(x, k, bias=None, stride=1, padding=0, groups=n*c)
    x = x.view(n, c, x.shape[2], x.shape[3])

    return x



def gen_kernel(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5*(scale_factor - 1) # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    #kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    # h[h < scipy.finfo(float).eps * h.max()] = 0
    ##################### CAREFUL, THIS MIGHT CREATE BUGS #######################
    h[h < 1e-8 * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha,1])])
    h1 = alpha/(alpha+1)
    h2 = (1-alpha)/(alpha+1)
    h = [[h1, h2, h1], [h2, -4/(alpha+1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)

"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""


def bicubic_degradation(x, sf=3):
    '''
    Args:
        x: HxWxC image, [0, 1]
        sf: down-scale factor

    Return:
        bicubicly downsampled LR image
    '''
    x = util.imresize_np(x, scale=1/sf)
    return x


def srmd_degradation(x, k, sf=3):
    ''' blur + bicubic downsampling

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2018learning,
          title={Learning a single convolutional super-resolution network for multiple degradations},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={3262--3271},
          year={2018}
        }
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')  # 'nearest' | 'mirror'
    x = bicubic_degradation(x, sf=sf)
    return x


def dpsr_degradation(x, k, sf=3):

    ''' bicubic downsampling + blur

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2019deep,
          title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={1671--1681},
          year={2019}
        }
    '''
    x = bicubic_degradation(x, sf=sf)
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    return x


def classical_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    #x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


def add_blur(img, sf=4, strength=1):
    wd2 = (4.0 + sf) * strength
    wd = (2.0 + 0.2*sf) * strength
    if random.random() < 0.5:
        l1 = wd2*random.random()
        l2 = wd2*random.random()
        k = anisotropic_Gaussian(ksize=2*random.randint(2,11)+3, theta=random.random()*np.pi, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', 2*random.randint(2,11)+3, wd*random.random())
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img

def add_smoothing(img, strength=1):
    wd2 = 20 * strength
    wd = 6 * strength
    r = random.random()
    if r < 1/3:
        l1 = wd2*random.random()
        l2 = wd2*random.random()
        k = anisotropic_Gaussian(ksize=2*random.randint(2,11)+3, theta=random.random()*np.pi, l1=l1, l2=l2)
    elif r < 2/3:
        k = fspecial('gaussian', 2*random.randint(2,11)+3, wd*random.random())
    else:
        s = random.randint(3, 30*strength)
        k = np.ones((s, s)) / s**2

    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img

def add_smoothing2(img, strength=1):
    wd2 = 30 * strength
    wd = 15 * strength
    r = random.random()
    if r < 1/3:
        l1 = wd2*random.random()
        l2 = wd2*random.random()
        k = anisotropic_Gaussian(ksize=int(max(l1, l2))*2+1, theta=random.random()*np.pi, l1=l1, l2=l2)
    elif r < 2/3:
        k = fspecial('gaussian', 2*random.randint(2,11)+3, wd2*random.random())
    else:
        s = random.randint(3, 30*strength)
        k = np.ones((s, s)) / s**2

    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img


def add_resize(img, sf=4, strength=1):
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = np.clip(random.uniform(0.5/sf/strength, 1), 0, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(img, (int(sf1*img.shape[1]), int(sf1*img.shape[0])), interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    return img


def add_Gaussian_noise(img, noise_level1=2, noise_level2=25, strength=1):
    noise_level = random.randint(noise_level1, noise_level2) * strength
    rnum = np.random.rand()
    if rnum > 0.6:   # add color Gaussian noise
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:            # add  noise
        L = noise_level2/255. * strength
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_speckle_noise(img, noise_level1=2, noise_level2=25, strength=1):
    noise_level = random.randint(noise_level1, noise_level2) * strength
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        img += img*np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img*np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2/255. * strength
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img*np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_Poisson_noise(img):
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = 10**(2*random.random()+2.0)  # [2, 4]
    if random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        img += noise_gray[:, :, np.newaxis]
    img = np.clip(img, 0.0, 1.0)
    return img


def add_JPEG_noise(img, quality_factor=None, bounds = (10, 95)):
    quality_factor = quality_factor or random.randint(*bounds)
    img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)
    return img


def random_crop(lq, hq, sf=4, lq_patchsize=64):
    h, w = lq.shape[:2]
    rnd_h = random.randint(0, h-lq_patchsize)
    rnd_w = random.randint(0, w-lq_patchsize)
    lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]

    rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
    hq = hq[rnd_h_H:rnd_h_H + lq_patchsize*sf, rnd_w_H:rnd_w_H + lq_patchsize*sf, :]
    return lq, hq

def add_enhance(img):
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    if random.random() < 0.5:
        enhance = ImageEnhance.Brightness(img)
    else:
        enhance = ImageEnhance.Contrast(img)

    img = enhance.enhance(random.random() + 0.5)
    img = np.array(img).astype(float) / 255
    return img

def degradation_ddrc(img):
    # enhance
    if random.random() < 0.5:
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        if random.random() < 0.5:
            enhance = ImageEnhance.Brightness(img)
        else:
            enhance = ImageEnhance.Contrast(img)

        img = enhance.enhance(random.random() + 0.5)
        img = np.array(img).astype(float) / 255

    # smoothing
    if random.random() < 0.5:
        k = random.randint(3, 15)
        if random.random() < 0.5:
            if k % 2 == 0:
                k += random.choice([-1, 1])
            img = cv2.GaussianBlur(img, (k,k), 0, 0)
        else:
            k = np.ones((k, k)) / k**2

            img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    # noise
    if random.random() < 0.3:
        level = random.randint(0, 50)
        noise = np.random.randn(*img.shape) * level / 255
        img = img + noise
        img = img.clip(0, 1)

    # jpeg
    if random.random() < 0.7:
        img = add_JPEG_noise(img, random.randint(10, 95))

    return img


def degradation_bsrgan(img, sf=4, lq_patchsize=64, isp_model=None, strength=1):
    """
    This is the degradation model of BSRGAN from the paper
    "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
    ----------
    img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    isp_model: camera ISP model

    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """
    isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25
    sf_ori = sf

    h1, w1 = img.shape[:2]
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
    h, w = img.shape[:2]

    if h < lq_patchsize*sf or w < lq_patchsize*sf:
        raise ValueError(f'img size ({h1}X{w1}) is too small!')

    hq = img.copy()

    if sf == 4 and random.random() < scale2_prob:   # downsample1
        if np.random.rand() < 0.5:
            img = cv2.resize(img, (int(1/2*img.shape[1]), int(1/2*img.shape[0])), interpolation=random.choice([1,2,3]))
        else:
            img = util.imresize_np(img, 1/2, True)
        img = np.clip(img, 0.0, 1.0)
        sf = 2

    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:  # keep downsample3 last
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:

        if i == 0:
            img = add_blur(img, sf=sf, strength=strength)

        elif i == 1:
            img = add_blur(img, sf=sf, strength=strength)
            
        elif i == 2:
            a, b = img.shape[1], img.shape[0]
            # downsample2
            if random.random() < 0.75:
                sf1 = random.uniform(1,2*sf) * strength
                img = cv2.resize(img, (int(1/sf1*img.shape[1]), int(1/sf1*img.shape[0])), interpolation=random.choice([1,2,3]))
            else:
                k = fspecial('gaussian', 25, random.uniform(0.1, 0.6*sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted/k_shifted.sum()  # blur with shifted kernel
                img = ndimage.filters.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
                img = img[0::sf, 0::sf, ...]  # nearest downsampling
            img = np.clip(img, 0.0, 1.0)
            pass

        elif i == 3:
            a, b = img.shape[1], img.shape[0]
            # downsample3
            img = cv2.resize(img, (int(a/sf/strength), int(b/sf/strength)), interpolation=random.choice([1,2,3]))
            img = np.clip(img, 0.0, 1.0)

        elif i == 4:
            # add Gaussian noise
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25, strength=strength)

        elif i == 5:
            # add JPEG noise
            if random.random() < jpeg_prob:
                img = add_JPEG_noise(img)

        elif i == 6:
            # add processed camera sensor noise
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)

    # add final JPEG compression noise
    img = add_JPEG_noise(img)

    # random crop
    #img, hq = random_crop(img, hq, sf_ori, lq_patchsize)
    img = cv2.resize(img, (w1, h1), interpolation=random.choice([1, 2, 3]))

    return img, hq



def degradation_bsrgan_plus(img, sf=4, shuffle_prob=0.5, use_sharp=True, lq_patchsize=64, isp_model=None, return_resized=True, strength=1):
    """
    This is an extended degradation model by combining
    the degradation models of BSRGAN and Real-ESRGAN
    ----------
    img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    use_shuffle: the degradation shuffle
    use_sharp: sharpening the img

    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """

    h1, w1 = img.shape[:2]
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
    h, w = img.shape[:2]

    if h < lq_patchsize*sf or w < lq_patchsize*sf:
        raise ValueError(f'img size ({h1}X{w1}) is too small!')

    if use_sharp:
        img = add_sharpening(img)
    hq = img.copy()

    if random.random() < shuffle_prob:
        shuffle_order = random.sample(range(13), 13)
    else:
        shuffle_order = list(range(13))
        # local shuffle for noise, JPEG is always the last one
        shuffle_order[2:6] = random.sample(shuffle_order[2:6], len(range(2, 6)))
        shuffle_order[9:13] = random.sample(shuffle_order[9:13], len(range(9, 13)))

    poisson_prob, speckle_prob, isp_prob = 0.1, 0.1, 0.1

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf, strength=strength)
        elif i == 1:
            img = add_resize(img, sf=sf, strength=strength)
        elif i == 2:
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25, strength=strength)
        elif i == 3:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img) # TODO: do we need a strength parameter here?
        elif i == 4:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img, strength=strength)
        elif i == 5:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        elif i == 6:
            img = add_JPEG_noise(img, bounds=(30, 95)) # TODO: strength?
        elif i == 7:
            img = add_blur(img, sf=sf, strength=strength)
        elif i == 8:
            img = add_resize(img, sf=sf, strength=strength)
        elif i == 9:
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25, strength=strength)
        elif i == 10:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 11:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img, strength=strength)
        elif i == 12:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        else:
            print('check the shuffle!')

    # resize to desired size
    #img = cv2.resize(img, (int(1/sf*hq.shape[1]), int(1/sf*hq.shape[0])), interpolation=random.choice([1, 2, 3]))

    # add final JPEG compression noise
    img = add_JPEG_noise(img, bounds=(30, 95))

    # random crop
    # img, hq = random_crop(img, hq, sf, lq_patchsize)

    if return_resized:
        img = cv2.resize(img, (w1, h1), interpolation=random.choice([1, 2, 3]))

    return img, hq

def degradation(img, shuffle_prob=0.5, strength=1, p=0.5, single=False):
    """
    Adjusted version of degradations_bsrgan_plus for deepfake detection
    """

    h1, w1 = img.shape[:2]

    num_options = 6 if single else 13

    if random.random() < shuffle_prob:
        shuffle_order = random.sample(range(num_options), num_options)
    else:
        shuffle_order = list(range(num_options))
        # local shuffle for noise, JPEG is always the last one
        shuffle_order[2:6] = random.sample(shuffle_order[2:6], len(range(2, 6)))
        if len(shuffle_order) > 6:
            shuffle_order[9:13] = random.sample(shuffle_order[9:13], len(range(9, 13)))

    poisson_prob, speckle_prob, isp_prob = 0.1, 0.1, 0.1

    sf = 2
    for i in shuffle_order:
        if i == 0:
            if random.random() < p:
                img = add_smoothing(img, strength=strength)
        elif i == 1:
            if random.random() < p:
                img = add_resize(img, sf=sf, strength=strength)
        elif i == 2:
            if random.random() < p:
                img = add_Gaussian_noise(img, noise_level1=2 if random.random() < 0.5 else 80, noise_level2=100, strength=strength)
        elif i == 3:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img) # TODO: do we need a strength parameter here?
        elif i == 4:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img, strength=strength)
        elif i == 5:
            if random.random() < p:
                add_enhance(img)
        elif i == 6:
            if random.random() < p:
                img = add_JPEG_noise(img) # TODO: strength?
        elif i == 7:
            if random.random() < p:
                img = add_smoothing(img, strength=strength)
        elif i == 8:
            if random.random() < p:
                img = add_resize(img, sf=sf, strength=strength)
        elif i == 9:
            if random.random() < p:
                img = add_Gaussian_noise(img, noise_level1=2 if random.random() < 0.5 else 75, noise_level2=100, strength=strength)
        elif i == 10:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 11:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img, strength=strength)
        elif i == 12:
            if random.random() < p:
                img = add_sharpening(img)
        else:
            print('check the shuffle!')

    # resize to desired size
    #img = cv2.resize(img, (int(1/sf*hq.shape[1]), int(1/sf*hq.shape[0])), interpolation=random.choice([1, 2, 3]))

    # add final JPEG compression noise
    img = add_JPEG_noise(img)

    # random crop
    # img, hq = random_crop(img, hq, sf, lq_patchsize)
    img = cv2.resize(img, (w1, h1), interpolation=random.choice([1, 2, 3]))

    return img



if __name__ == '__main__':
    img = util.imread_uint('utils/test.png', 3)
    img = util.uint2single(img)
    sf = 4
    
    for i in range(20):
        img_lq, img_hq = degradation_bsrgan(img, sf=sf, lq_patchsize=72)
        print(i)
        lq_nearest =  cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
        img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
        util.imsave(img_concat, str(i)+'.png')

#    for i in range(10):
#        img_lq, img_hq = degradation_bsrgan_plus(img, sf=sf, shuffle_prob=0.1, use_sharp=True, lq_patchsize=64)
#        print(i)
#        lq_nearest =  cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
#        img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
#        util.imsave(img_concat, str(i)+'.png')

#    run utils/utils_blindsr.py
