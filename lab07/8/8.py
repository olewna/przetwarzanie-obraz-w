import cv2
import numpy as np

# Ścieżka do obrazu
image_path = "lab07/initial data/DoubleDuckHongkong.png"
img = cv2.imread(image_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = img

# Wygładzenie filtrem Gaussa
blurred = cv2.GaussianBlur(img_rgb, (5, 5), sigmaX=2, sigmaY=2)
cv2.imwrite("lab07/8/8a.png", blurred)

# Filtry Sobela
# h_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
# h_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

h_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
h_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

# Gradienty kolorów
# R_x = ndimage.filters.convolve(blurred[:,:,0].astype(np.float32) /255, h_x)
Rx = cv2.filter2D(blurred[:, :, 0].astype(np.float32) / 255.0, -1, h_x)
Gx = cv2.filter2D(blurred[:, :, 1].astype(np.float32) / 255.0, -1, h_x)
Bx = cv2.filter2D(blurred[:, :, 2].astype(np.float32) / 255.0, -1, h_x)
cv2.imshow("Blue", Rx)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("lab07/8/8b_Rx.png", Rx*255)
cv2.imwrite("lab07/8/8b_Gx.png", Gx*255)
cv2.imwrite("lab07/8/8b_Bx.png", Bx*255)

Ry = cv2.filter2D(blurred[:, :, 0].astype(np.float32) / 255.0, -1, h_y)
Gy = cv2.filter2D(blurred[:, :, 1].astype(np.float32) / 255.0, -1, h_y)
By = cv2.filter2D(blurred[:, :, 2].astype(np.float32) / 255.0, -1, h_y)

cv2.imwrite("lab07/8/8b_Ry.png", Ry*255)
cv2.imwrite("lab07/8/8b_Gy.png", Gy*255)
cv2.imwrite("lab07/8/8b_By.png", By*255)

# Gradient kierunkowy (norma L∞)
g_x = np.maximum.reduce([np.abs(Rx), np.abs(Gx), np.abs(Bx)])
g_y = np.maximum.reduce([np.abs(Ry), np.abs(Gy), np.abs(By)])
cv2.imwrite("lab07/8/8c_gx.png", g_x*255)
cv2.imwrite("lab07/8/8c_gy.png", g_y*255)

# Gradient i kierunek
G = np.hypot(g_x, g_y) 
theta = np.rad2deg(np.arctan2(g_y, g_x))
theta[theta < 0] += 180  # Zakres [0, 180]
theta = np.round(theta, 2)
cv2.imwrite("lab07/8/8c_gradient.png", G*255)
cv2.imwrite("lab07/8/8c_kat.png", theta)

directions = np.zeros_like(theta, dtype=np.float32)
directions[(theta >= 0) & (theta < 22.5)] = 0
directions[(theta >= 157.5) & (theta <= 180)] = 0
directions[(theta >= 22.5) & (theta < 67.5)] = 45
directions[(theta >= 67.5) & (theta < 112.5)] = 90
directions[(theta >= 112.5) & (theta < 157.5)] = 135

# 7. Koloryzacja kierunków gradientu
colored_directions = np.zeros((*directions.shape, 3), dtype=np.float32)
colored_directions[directions == 0] = (0, 255, 255)   # żółty - 0°
colored_directions[directions == 45] = (255, 0, 0)    # czerwony - 45°
colored_directions[directions == 90] = (0, 0, 255)    # niebieski - 90°
colored_directions[directions == 135] = (0, 255, 0)

# print(np.sum((theta >= 0) & (theta < 22.5)))
# print(np.sum((theta >= 67.5) & (theta < 112.5)))
# print(np.sum((theta >= 112.5) & (theta < 157.5)))
# print(np.sum((theta >= 22.5) & (theta < 67.5)))
cv2.imwrite("lab07/8/8d.png", colored_directions)

# Funkcja Non-Maximum Suppression
def non_maximum_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.float32)
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            angle = theta[i, j]

            q = 255
            r = 255

            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            elif 22.5 <= angle < 67.5:
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            elif 67.5 <= angle < 112.5:
                q = G[i+1, j]
                r = G[i-1, j]
            elif 112.5 <= angle < 157.5:
                q = G[i-1, j-1]
                r = G[i+1, j+1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0
    return Z

# Zastosowanie NMS
nms_result = non_maximum_suppression(G, theta)
cv2.imwrite("lab07/8/8e.png", nms_result*255) # nie wiem czy tak powinno być XD ale wyglada dobrze pozdro

# nms_uint8 = np.clip(nms_result, 0, 255).astype(np.uint8)
# print(nms_uint8)

def otsu(img):
    # Histogram H(i) – Eq. (1)
    H, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))

    # Rozmiar obrazu
    total = img.size

    # Skumulowana funkcja rozkładu C(i) – Eq. (2)
    C = np.cumsum(H) / total

    # Średnia intensywność μ – Eq. (3)
    intensities = np.arange(256)
    mu = np.sum(intensities * H) / total

    # Wstępne wartości pomocnicze
    mu_t = np.cumsum(intensities * H) / total

    # Eq. (5): P0(T), P1(T)
    P0 = C
    P1 = 1 - C

    # Eq. (6), (7): m0(T), m1(T)
    m0 = np.zeros(256)
    m1 = np.zeros(256)

    nonzero_P0 = P0 > 0
    nonzero_P1 = P1 > 0

    m0[nonzero_P0] = mu_t[nonzero_P0] / P0[nonzero_P0]
    m1[nonzero_P1] = (mu - mu_t[nonzero_P1]) / P1[nonzero_P1]

    # Eq. (4): wariancja między klasami
    var_between = P0 * P1 * (m0 - m1) ** 2

    # Eq. (8): T_opt jako T maksymalizujące wariancję między klasami
    T_opt = int(np.argmax(var_between))

    return T_opt

T2 = otsu(nms_result*255)
# T2 = 150
T1 = 0.5 * T2
print(f"Próg Otsu: {T2}")
# cv2.imshow("RX", nms_result)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Funkcja histerezy
def hysteresis_threshold(img, t1, t2):
    strong = 255
    weak = 75

    output = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= t2)
    weak_i, weak_j = np.where((img >= t1) & (img < t2))

    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak

    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if output[i, j] == weak:
                if np.any(output[i-1:i+2, j-1:j+2] == strong):
                    output[i, j] = strong
                else:
                    output[i, j] = 0
    return output

# Zastosowanie histerezy
canny_result = hysteresis_threshold(nms_result*255, T1, T2)
cv2.imwrite("lab07/8/8f.png", canny_result)