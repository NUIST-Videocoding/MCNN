# -*- coding: utf-8 -*-
import os
import sys
import csv
import argparse
from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument('--image_path', default=r'E:\Datasets\OpenImageV6\seg_val\data', metavar="DIR",
                        help='Original images path')
    parser.add_argument('--bin_path', default=r'E:\whh\VCM\W-MSFC\save\model_9_8_no_ar_mse_cossim_seg\0.25\results_bin\1115000', metavar="DIR",
                        help='Bin path')
    return parser


args = get_parser().parse_args()
bin_path = args.bin_path
bin_list = os.listdir(bin_path)

img_path = args.image_path
img_list = os.listdir(img_path)

if not os.path.exists(r'E:\whh\VCM\W-MSFC\save\model_9_8_no_ar_mse_cossim_seg\0.125'):
    os.mkdir(r'E:\whh\VCM\W-MSFC\save\model_9_8_no_ar_mse_cossim_seg\0.125')
f = open(r'E:\whh\VCM\W-MSFC\save\model_9_8_no_ar_mse_cossim_seg\0.25\bpp_info_1115000.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["Number", "Image_Name", "Width", "Height", "Pixel", "Bin_Size"])
count = 0
pixel_total = 0
bit_total = 0
for bin in bin_list:
    count += 1
    name = bin.split('.')[0]
    img_id = name + '.jpg'
    img_name = os.path.join(img_path, img_id)
    img = Image.open(img_name)
    W, H = img.size
    pixel = W * H
    bit = os.path.getsize(os.path.join(bin_path, bin))
    pixel_total += pixel
    bit_total += bit
    csv_writer.writerow([count, img_id, W, H, W * H, bit])
bpp_rate = bit_total * 8 / pixel_total
csv_writer.writerow(["Total", "", "", "", pixel_total, bit_total])
csv_writer.writerow(["Bpp_rate", bpp_rate])
f.close()
print("Calculate {} images:\n\ttotal pixels: {}\n\ttotal bin size: {}\n\tbpp rate: {}"
      .format(count, pixel_total, bit_total, bpp_rate))
print("You can find details in output folder.\nDone!")
