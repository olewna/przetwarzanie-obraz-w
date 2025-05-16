import numpy as np
from PIL import Image

def global_contrast(image):
    L_max = np.max(image)
    L_min = np.min(image)
    
    if image.dtype == np.uint8:
        g_range = 255
    else:
        g_range = L_max - L_min
    
    if g_range == 0:
        return 0.0
    
    contrast = (L_max - L_min) / g_range
    print(f'Global Contrast: {contrast:.4f}')
    return contrast

def local_contrast(image):
    M, N = image.shape

    padded_image = np.full((M+2, N+2), -1, dtype=np.int32)
    padded_image[1:M+1, 1:N+1] = image

    total_difference = 0
    for m in range(1, M+1):
        for n in range(1, N+1):
            neighborhood = padded_image[m-1:m+2, n-1:n+2]
            neighborhood = np.where(neighborhood == -1, padded_image[m, n], neighborhood)
            mean_nb = (np.sum(neighborhood) - padded_image[m, n]) / 8  
            total_difference += abs(padded_image[m, n] - mean_nb)
    
    contrast_local = total_difference / (M * N)
    print(f'Local Contrast: {contrast_local:.4f}')
    return contrast_local

image01 = 'szerszenA.png'
image02 = 'szerszenB.png'
image03 = 'szerszenC.png'

input_image01 = Image.open(image01).convert('L')
img01 = np.array(input_image01, dtype=np.uint8)
global_contrast(img01)
local_contrast(img01)

input_image02 = Image.open(image02).convert('L')
img02 = np.array(input_image02, dtype=np.uint8)
global_contrast(img02)
local_contrast(img02)

input_image03 = Image.open(image03).convert('L')
img03 = np.array(input_image03, dtype=np.uint8)
global_contrast(img03)
local_contrast(img03)