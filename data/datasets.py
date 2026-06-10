import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage import gaussian_filter
from torchvision.transforms import InterpolationMode
import torch.nn.functional as nnF
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Frequency Attenuation Augment ──────────────────────────────────
class FrequencyAttenuate:
    """Randomly weaken the highest-frequency NPR signal during training.

    By 2×2 average-pooling and bilinear upsampling, the f≈0.5 (period-2)
    component is suppressed while f≈0.25 (period-4) is largely preserved.
    This forces the model to rely on multi-band NPR cues instead of a single
    fragile frequency, improving robustness to JPEG / blur without harming
    cross-generator generalisation.

    Reference: Sec 4.4 of the proposed Multi-Scale Edge-Guided NPR.
    """

    def __init__(self, prob=0.3, alpha=0.3):
        self.prob = prob
        self.alpha = alpha

    def __call__(self, tensor):
        """Args:
            tensor: [C, H, W] normalised image (ToTensor + Normalize already applied).
        Returns:
            tensor: same shape, with high-freq possibly attenuated.
        """
        if self.prob <= 0 or random() >= self.prob:
            return tensor

        C, H, W = tensor.shape
        x = tensor.unsqueeze(0)                        # [1, C, H, W]

        # 2×2 avg pool → highest frequencies averaged out
        down = nnF.avg_pool2d(x, 2, 2)
        up = nnF.interpolate(down, size=(H, W), mode='bilinear',
                             align_corners=False)

        # alpha * smooth + (1-alpha) * original
        result = x + self.alpha * (up - x)              # [1, C, H, W]
        return result.squeeze(0)

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))

    transform_list = [
        rz_func,
        transforms.Lambda(lambda img: data_augment(img, opt)) if opt.isTrain and opt.data_aug
        else transforms.Lambda(lambda img: img),
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    # ── optional frequency attenuation for training ──
    freq_prob = getattr(opt, 'freq_aug_prob', 0.0)
    freq_alpha = getattr(opt, 'freq_aug_alpha', 0.3)
    if opt.isTrain and freq_prob > 0:
        transform_list.append(FrequencyAttenuate(prob=freq_prob, alpha=freq_alpha))

    dset = datasets.ImageFolder(root, transforms.Compose(transform_list))
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


# rz_dict = {'bilinear': Image.BILINEAR,
           # 'bicubic': Image.BICUBIC,
           # 'lanczos': Image.LANCZOS,
           # 'nearest': Image.NEAREST}
rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize,opt.loadSize), interpolation=rz_dict[interp])
