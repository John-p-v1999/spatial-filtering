# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import cv2
import numpy as np

def addNoise(image, SNR):
    image = np.atleast_3d(image)
    noisyImage = image.copy()
    
    X, Y, Z = image.shape
    mask = np.random.choice((0, 1, 2), size = (X, Y), p = [SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    
    for x in range(X):
        for y in range(Y):
            if mask[x][y] ==  1:
                noisyImage[x, y, :] = 0
            elif mask[x][y] == 2:
                noisyImage[x, y, :] = 255
    
    return noisyImage

def medianFilter(image, kernel_size):
    assert kernel_size % 2 != 0, 'Kernel of the median filter should be an odd number.'
    l = kernel_size // 2
    image = np.atleast_3d(image)
    X, Y, Z = image.shape
    filteredImage = image.copy()
    for z in range(Z):
        for x in range(l, X - l):
            for y in range(l, Y - l):
                array = image[x : x+kernel_size, y : y + kernel_size, z].reshape(-1)
                filteredImage[x, y, z] = np.median(array)
    
    image = np.squeeze(image)
    filteredImage = np.squeeze(filteredImage)
    return filteredImage.astype('uint8')

def plot(images, captions):
    n_col = 2
    n_row = len(images) // 2
    plt.figure(figsize = (4 * n_col, 4 * n_row))
    
    for i in range(len(images)):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i])
        plt.title(captions[i])
        plt.xticks(()), plt.yticks(())
        plt.show()
        
img = cv2.imread(r'C:\Users\asus\Desktop\sem7\computer vision\spacial filtering\lena.bmp')
nimg = addNoise(img, 0.6)

images = [img, nimg]
captions = ['Original Image', 'Noisy Image']
for i in range(3, 10, 2):
    cimg = medianFilter(nimg, i)
    images.append(cimg)
    captions.append(f'Corrected Image\nMedian Filter - {i}x{i}')
    
plot(images, captions)