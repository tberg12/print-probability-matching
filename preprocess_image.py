"""
Preprocess image like make_twin_dataset_from_splits.py
"""
from PIL import Image
from torchvision import transforms
from make_twin_dataset_from_splits import Binarizer, SquarePad
import argparse
from pathlib import Path
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--unpadded_image_size', type=int, default=112, help='Size of output images')
    parser.add_argument('--padded_image_size', type=int, default=224, help='Size of output images')
    args = parser.parse_args()

    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((args.unpadded_image_size, args.unpadded_image_size)),
        transforms.Grayscale(num_output_channels=1),
        Binarizer(),
        transforms.Pad((args.padded_image_size - args.unpadded_image_size) // 2, fill=255, padding_mode='constant'),
    ])

    binarizer = Binarizer()
    square_pad = SquarePad()
    img = Image.open(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.array(transform(img))).save(args.output)
