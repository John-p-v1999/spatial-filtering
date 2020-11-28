# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import cv2
import numpy as np

class GaussianFilter:
    
    def __init__(self, dimension, sigma = 1):
        assert dimension % 2 != 0, "Kernel dimension should be an odd number";
        self.dimension = dimension
        self.kernel = np.zeros((dimension, dimension, 1))
        for i in range(dimension):
            x = i - dimension // 2
            for j  in range(dimension):
                y = j - dimension // 2
                self.kernel[i, j, :] = np.exp(-(x**2 + y**2) / (2 * (sigma ** 2))) / (2 * np.pi * (sigma ** 2))
                
        self.kernel = self.kernel / np.sum(self.kernel)
        
    def _padImage(self, image):
        N = self.dimension // 2
        image = np.atleast_3d(image)
        X, Y, Z = image.shape
        padded = np.zeros((X + 2*N, Y + 2*N, Z))
        padded[N : X + N, N : Y + N, :] = image
        return np.squeeze(padded)
    
    def filterImage(self, image):
        image = np.atleast_3d(image)
        X, Y, Z = image.shape
        image = np.squeeze(image)
        assert self.dimension <= min(X, Y), "Image is too small for the filter."
        paddedImage = self._padImage(image)
        filteredImage = np.zeros((X, Y, Z))
        for x in range(X):
            for y in range(Y):
                rect = paddedImage[x : x + self.dimension, y : y + self.dimension, :]
                filteredImage[x, y, :] = np.sum(np.multiply(rect, self.kernel), axis = (0, 1))
        
        return filteredImage.astype('uint8')
    
def plot(image, mask_sizes, stddev):
    n_col = 2
    n_row = len(mask_sizes)
    plt.figure(figsize = (4 * n_col, 4 * n_row))
    i = 0
    for size in mask_sizes:
        gf = GaussianFilter(size, stddev)
        filtered = gf.filterImage(image)
        filtered_cv = cv2.GaussianBlur(image, (size, size), stddev, sigmaY=stddev, borderType = cv2.BORDER_CONSTANT)
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(filtered, cmap = plt.cm.gray)
        plt.title(f'Mask size = {size}')
        plt.xticks(()), plt.yticks(())
        plt.subplot(n_row, n_col, i+2)
        plt.show()
        i += 2
        
if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread(r'C:\Users\asus\Desktop\sem7\computer vision\spacial filtering\lena_color.bmp'), cv2.COLOR_RGB2BGR)
    plot(img, mask_sizes = [3, 5, 15, 31], stddev = 5)