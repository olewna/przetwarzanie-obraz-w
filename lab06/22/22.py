import numpy as np
import cv2

# image_path = 'lab06/initial data/SzukanieJedynek.png'
image_path = 'lab06/22/rotated.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def otsu(img):
    H, _ = np.histogram(img.ravel(), bins=256, range=(-1, 256))

    total = img.size

    C = np.cumsum(H) / total

    intensities = np.arange(256)
    mu = np.sum(intensities * H) / total

    mu_t = np.cumsum(intensities * H) / total

    P0 = C
    P1 = 1 - C

    m0 = np.zeros(256)
    m1 = np.zeros(256)

    nonzero_P0 = P0 > 0
    nonzero_P1 = P1 > 0

    m0[nonzero_P0] = mu_t[nonzero_P0] / P0[nonzero_P0]
    m1[nonzero_P1] = (mu - mu_t[nonzero_P1]) / P1[nonzero_P1]

    var_between = P0 * P1 * (m0 - m1) ** 2

    T_opt = int(np.argmax(var_between))

    return T_opt

prog_otsu = otsu(img)
print(prog_otsu)

_, threshold_normal = cv2.threshold(img, prog_otsu, 255, cv2.THRESH_BINARY)
cv2.imwrite("lab06/22/22_threshold_normal.png", threshold_normal)
_, thresholded = cv2.threshold(img, prog_otsu, 255, cv2.THRESH_BINARY_INV)
# cv2.imwrite("lab06/22/22_threshold2.png", thresholded)

fixing_holes = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, fixing_holes)
# cv2.imwrite("lab06/22/22_closed.png", closed)

# jedynka = np.array([[-1, -1,  -1, -1, -1],
#                     [-1, -1,  0, 0, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, -1],
#                     [-1, 0,  1, 1, 1],
#                     [-1, 0,  0, 0, 0],
#                     [-1, -1,  -1, -1, -1]], dtype=np.int8)

jedynka2 = np.array([
                    [-1,   -1,   0,   0,    -1], # 0 czubek
                    [-1,    1,   1,   1,    -1], # 1
                    [-1,    0,   1,   1,    -1], # 2
                    [-1,    0,   1,   1,    -1], # 3
                    [-1,   -1,   1,   1,    -1], # 4
                    [-1,   -1,   1,   1,    -1], # 5
                    [-1,   -1,   1,   1,    -1], # 6
                    [-1,   -1,   1,   1,    -1], # 7
                    [-1,   -1,   1,   1,    -1], # 8
                    [-1,   -1,   1,   1,    -1], # 9
                    [-1,   -1,   1,   1,    -1], # 10
                    [-1,    0,   1,   1,     0], # 11
                    [-1,    0,   1,   1,     0], # 12 podstawa
                    [-1,   -1,  -1,  -1,    -1,]  # 13
                    ], dtype=np.int8)

# jedynka2 = np.flipud(np.fliplr(jedynka2))

idealna_jedynka = np.array([
                    [0, 0,  0], # 13 
                    [1, 1,  1], # 12 podstawa
                    [0, 1,  0], # 11
                    [0, 1,  0], # 10
                    [0, 1,  0], # 9
                    [0, 1,  0], # 8
                    [0, 1,  0], # 7
                    [0, 1,  0], # 6
                    [0, 1,  0], # 5
                    [0, 1,  0], # 4
                    [0, 1,  0], # 3
                    [0, 1,  0], # 2
                    [0, 1,  1], # 1
                    [0, 1,  0]  # 0
                    ], dtype=np.uint8)

binary_01 = (closed/255).astype(np.uint8)

hitmiss = cv2.morphologyEx(binary_01, cv2.MORPH_HITMISS, jedynka2)
cv2.imwrite("lab06/22/22_hitmiss.png", hitmiss*255)

dilated_image = cv2.dilate(hitmiss*255, idealna_jedynka, iterations=1)
cv2.imwrite("lab06/22/22_dilated.png", dilated_image)

inverted = cv2.bitwise_not(dilated_image)
cv2.imwrite("lab06/22/22_inverted.png", inverted)

without_ones = np.where(inverted == 0, 255, threshold_normal).astype(np.uint8)
cv2.imwrite("lab06/22/22_without_ones.png", without_ones)

# cv2.imshow("xd", without_ones)
# cv2.waitKey(0)
# cv2.destroyAllWindows()