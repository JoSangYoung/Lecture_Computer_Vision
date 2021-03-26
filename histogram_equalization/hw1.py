import cv2
import numpy as np
import matplotlib.pyplot as plt

def Calculate_Histogram(img):
    histogram = np.zeros([256], dtype=int)
    cumulative_sum = np.zeros([256], dtype=int)
    for i in img:
        histogram[i] = histogram[i] + 1

    cumulative_sum[0] = histogram[0]
    for i in range(1, 256):
        cumulative_sum[i] = cumulative_sum[i-1] + histogram[i]

    return cumulative_sum

def HE(img, cdf, width, height):
    cdf_max = np.max(cdf)
    cdf_min = np.min(cdf)
    he_img = img

    he_img[:,:] = np.array( ((cdf[img[:,:]] - cdf_min) / (cdf_max - cdf_min)) * 255, dtype=int)
    return he_img

img = cv2.imread("./city.png",cv2.IMREAD_GRAYSCALE)
width = img.shape[1]
height = img.shape[0]

cdf = Calculate_Histogram(img)
he_img = HE(img, cdf, width, height)

cv2.imwrite('HE_city.png',he_img)
