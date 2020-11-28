# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:20:57 2020

@author: jCube
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

from GaussianFilter import GaussianFilter

def unsharpMask(image, blurredImage):
    mask = image - blurredImage
    return (image + mask).astype('uint8')

def highboostFilter(image, blurredImage, k):
    mask = image - blurredImage
    return (image + k * mask).astype('uint8')

def plot(image, mask_sizes, stddev):
    n_col = 3
    n_row = len(mask_sizes)
    plt.figure(figsize = (3.5 * n_col, 4 * n_row))
    i = 0
    for size in mask_sizes:
        gf = GaussianFilter(size, stddev)
        filtered = gf.filterImage(image)
        
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(image, cmap = plt.cm.gray)
        plt.title(f'Actual Image')
        plt.xticks(()), plt.yticks(())
        plt.subplot(n_row, n_col, i+2)
        plt.imshow(unsharpMask(image, filtered), cmap = plt.cm.gray)
        plt.title(f'Unsharp Masking\n{size}x{size}')
        plt.xticks(()), plt.yticks(())
        plt.subplot(n_row, n_col, i+3)
        plt.imshow(highboostFilter(image, filtered, 3), cmap = plt.cm.gray)
        plt.title(f'Highboost Filtering\n{size}x{size}')
        plt.xticks(()), plt.yticks(())
        i += 3
    plt.show()
        
img = cv2.imread(r'C:\Users\asus\Desktop\sem7\computer vision\spacial filtering\lena.bmp')
plot(img, [3, 5, 7], 1)