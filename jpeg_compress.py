import os
import sys
import cv2
import argparse
from io import BytesIO
from PIL import Image
import numpy as np


def cv2_compress(img, quality):
    """Compress image via cv2 JPEG encode/decode.
    
    Args:
        img: numpy array (H, W, C) in RGB
        quality: int, JPEG quality [0, 100]
    Returns:
        compressed img in RGB
    """
    img_bgr = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_compress(img, quality):
    """Compress image via PIL JPEG encode/decode.
    
    Args:
        img: numpy array (H, W, C) in RGB
        quality: int, JPEG quality [1, 95] (PIL range)
    Returns:
        compressed img in RGB
    """
    out = BytesIO()
    pil_img = Image.fromarray(img)
    pil_img.save(out, format='jpeg', quality=quality)
    pil_img = Image.open(out)
    compressed = np.array(pil_img)
    out.close()
    return compressed


jpeg_methods = {'cv2': cv2_compress, 'pil': pil_compress}


def compress_image(img, quality, method='cv2'):
    return jpeg_methods[method](img, quality)


def compress_folder(input_dir, output_dir, quality, method='cv2'):
    """Apply JPEG compression to all images in a folder tree.
    
    Maintains the original folder structure under output_dir.
    Supports common image extensions: png, jpg, jpeg, bmp, webp.
    """
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    count = 0
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, input_dir)
            dst = os.path.join(output_dir, rel)

            img = cv2.imread(src)
            if img is None:
                print(f"Skipping unreadable: {src}")
                continue
            img = img[:, :, ::-1]  # BGR -> RGB

            compressed = compress_image(img, quality, method)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, compressed[:, :, ::-1])  # RGB -> BGR
            count += 1
    print(f"Compressed {count} images (quality={quality}, method={method})")
    print(f"  {input_dir}  ->  {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True, help='path to input image folder')
    parser.add_argument('--output_dir', required=True, help='path to output folder')
    parser.add_argument('--quality',    required=True, type=int, help='JPEG quality [0, 100]')
    parser.add_argument('--method',     default='cv2', choices=['cv2', 'pil'], help='backend')
    opt = parser.parse_args()
    compress_folder(opt.input_dir, opt.output_dir, opt.quality, opt.method)
