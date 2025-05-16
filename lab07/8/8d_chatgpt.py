import cv2
import numpy as np

# Wczytanie obrazu
img = cv2.imread("lab07/initial data/DoubleDuckHongkong.png")
img_rgb =  img#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Wygładzenie filtrem Gaussa z sigma = 2
blurred = cv2.GaussianBlur(img_rgb, (5, 5), sigmaX=2, sigmaY=2)

# 2. Zdefiniowanie filtrów Sobela
h_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
h_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# 3. Obliczenie gradientów R, G, B w kierunkach X i Y
Rx = cv2.filter2D(blurred[:, :, 0], -1, h_x)
Gx = cv2.filter2D(blurred[:, :, 1], -1, h_x)
Bx = cv2.filter2D(blurred[:, :, 2], -1, h_x)

Ry = cv2.filter2D(blurred[:, :, 0], -1, h_y)
Gy = cv2.filter2D(blurred[:, :, 1], -1, h_y)
By = cv2.filter2D(blurred[:, :, 2], -1, h_y)

# 4. Obliczenie gradientów kierunkowych g_x, g_y (norma L-infinity - maksimum kanału)
g_x = np.maximum.reduce([np.abs(Rx), np.abs(Gx), np.abs(Bx)])
g_y = np.maximum.reduce([np.abs(Ry), np.abs(Gy), np.abs(By)])

# 5. Obliczenie wartości gradientu i kąta
G = np.hypot(g_x, g_y)
theta = np.rad2deg(np.arctan2(g_y, g_x))
theta[theta < 0] += 180  # zakres [0, 180)

# 6. Kwantyzacja kierunku gradientu
directions = np.zeros_like(theta, dtype=np.uint8)
directions[(theta >= 0) & (theta < 22.5)] = 0
directions[(theta >= 157.5) & (theta <= 180)] = 0
directions[(theta >= 22.5) & (theta < 67.5)] = 45
directions[(theta >= 67.5) & (theta < 112.5)] = 90
directions[(theta >= 112.5) & (theta < 157.5)] = 135

# 7. Koloryzacja kierunków gradientu
colored_directions = np.zeros((*directions.shape, 3), dtype=np.uint8)
colored_directions[directions == 0] = (255, 255, 0)   # żółty - 0°
colored_directions[directions == 45] = (255, 0, 0)    # czerwony - 45°
colored_directions[directions == 90] = (0, 0, 255)    # niebieski - 90°
colored_directions[directions == 135] = (0, 255, 0)   # zielony - 135°

colored_directions_bgr = cv2.cvtColor(colored_directions, cv2.COLOR_RGB2BGR)
cv2.imwrite("lab07/8/chat_gradient.png", colored_directions_bgr)
