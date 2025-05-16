import cv2
import numpy as np

G_x = cv2.imread("lab07/8/8c_gx.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
G_y = cv2.imread("lab07/8/8c_gy.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
# gradient_image = cv2.imread("lab07/8/8c_gradient.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)

def gradient_i_theta(Gx, Gy):
    theta = np.arctan2(Gy, Gx)

    G = np.hypot(Gx, Gy)
    G = G / G.max() * 255

    return (G,theta)

def non_max_suppression(img, thetaRadian):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = thetaRadian * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

def otsu(image):
    gray = image.copy()
    if gray.max() > 1:
        gray = np.uint8(gray / gray.max() * 255)

    # Oblicz histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    print(hist)

    total_pixels = gray.shape[0] * gray.shape[1]
    sum_total = np.dot(np.arange(256), hist)

    sum_background = 0.0
    weight_background = 0.0
    max_variance = 0.0
    threshold = 0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # Obliczenie wariancji międzyklasowej
        between_class_variance = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )

        # Maksymalizacja wariancji międzyklasowej
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            threshold = t

    return threshold

def threshold(img, lowThresholdRatio=0.50, highThresholdRatio=0.90):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

gradient_image, theta = gradient_i_theta(G_x, G_y)

non_maximum = non_max_suppression(gradient_image, theta)
# cv2.imwrite("lab07/8/8e.png", non_maximum)

# print(otsu(non_maximum))
_, weak, strong = threshold(non_maximum)
result_image = hysteresis(non_maximum, 0, strong)

cv2.imwrite("lab07/8/8f.png", result_image)