import numpy as np
import cv2

image_path = 'lab06/initial data/StoLat.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def otsu(img):
    H, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))

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

_, thresholded = cv2.threshold(img, prog_otsu, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("lab06/21/21_threshold.png", thresholded)

vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
# print(vertical_structure)

eroded_image = cv2.erode(thresholded, vertical_structure, iterations=1)
cv2.imwrite("lab06/21/21_eroded.png", eroded_image)
takty = np.sum(eroded_image == 255)
# print(takty)

dilated_image = cv2.dilate(eroded_image, vertical_structure, iterations=1)
cv2.imwrite("lab06/21/21_dilated.png", dilated_image)

inverted = cv2.bitwise_not(dilated_image)
cv2.imwrite("lab06/21/21_inverted.png", inverted)

# cv2.imshow("xd", dilated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()