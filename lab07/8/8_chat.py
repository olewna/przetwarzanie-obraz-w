# Ponowne załadowanie obrazu i bibliotek po resecie środowiska
import cv2
import numpy as np

# Ścieżka do obrazu
image_path = "lab07/initial data/DoubleDuckHongkong.png"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Wygładzenie filtrem Gaussa
blurred = cv2.GaussianBlur(img_rgb, (5, 5), sigmaX=2, sigmaY=2)

# Filtry Sobela
h_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
h_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# Gradienty kolorów
Rx = cv2.filter2D(blurred[:, :, 0], -1, h_x)
Gx = cv2.filter2D(blurred[:, :, 1], -1, h_x)
Bx = cv2.filter2D(blurred[:, :, 2], -1, h_x)

Ry = cv2.filter2D(blurred[:, :, 0], -1, h_y)
Gy = cv2.filter2D(blurred[:, :, 1], -1, h_y)
By = cv2.filter2D(blurred[:, :, 2], -1, h_y)

# Gradient kierunkowy (norma L∞)
g_x = np.maximum.reduce([np.abs(Rx), np.abs(Gx), np.abs(Bx)])
g_y = np.maximum.reduce([np.abs(Ry), np.abs(Gy), np.abs(By)])

# Gradient i kierunek
G = np.hypot(g_x, g_y)
theta = np.rad2deg(np.arctan2(g_y, g_x))
theta[theta < 0] += 180  # Zakres [0, 180]

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
nms_result_uint8 = np.uint8(np.clip(nms_result, 0, 255))
cv2.imwrite("lab07/8/8e.png", nms_result_uint8)

# Próg Otsu
_, otsu_thresh = cv2.threshold(nms_result_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
T2 = otsu_thresh
T1 = 0.5 * T2

# Funkcja histerezy
def hysteresis_threshold(img, t1, t2):
    strong = 255
    weak = 75

    output = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= t2)
    weak_i, weak_j = np.where((img >= t2) & (img < t1))

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
canny_result = hysteresis_threshold(nms_result_uint8, T1, T2)
cv2.imwrite("/mnt/data/Canny_result.png", canny_result)

"/mnt/data/NMS_result.png", "/mnt/data/Canny_result.png", T2
