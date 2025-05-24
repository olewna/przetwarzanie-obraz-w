import cv2
import numpy as np

# Ścieżka do obrazu
image_path = "lab07/initial data/muszla.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# print(img)
img = np.array(img).astype(np.float32) /255
# print(img)


# Wygładzenie filtrem Gaussa
blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=2, sigmaY=2)
# cv2.imwrite("lab07/11/11a.png", blurred)

# h_xx = (1 0 -2 0 1)
# h_yy = (
#        1
#        0
#        -2
#        0
#        1
# )
# h_xy = (
#        -1 0 1
#         0 0 0
#         1 0 -1
# )

hxx = np.array([1,0,-2,0,1], dtype=np.float32)
hyy = np.array([[1],[0],[-2],[0],[1]], dtype=np.float32)
hxy = np.array([[-1,0,1],[0,0,0],[1,0,-1]], dtype=np.float32)

gxx = cv2.filter2D(blurred.astype(np.float32), -1, hxx)
gyy = cv2.filter2D(blurred.astype(np.float32), -1, hyy)
gxy = cv2.filter2D(blurred.astype(np.float32), -1, hxy)
# cv2.imwrite("lab07/11/11b_gxx.png", gxx)
# cv2.imwrite("lab07/11/11b_gyy.png", gyy)
# cv2.imwrite("lab07/11/11b_gxy.png", gxy)

def Newton(n, k):
    Wynik = 1

    for i in range(1, k + 1):
        Wynik = Wynik * (n - i + 1) / i
    return Wynik

def Hesse(gxx, gyy, gxy):
    M, N = gxx.shape
    Z = np.zeros((M, N), dtype=np.float32)
    theta = np.zeros((M,N), dtype=np.float32)

    for m in range(1, M-1):
        for n in range(1, N-1):
            H = np.array([[gxx[m][n], gxy[m][n]], [gxy[m][n], gyy[m][n]]])
            k1 = ((1/2)*(H[0][0] + H[1][1])) - ((1/4) * (np.sqrt((np.power(H[0][0] + H[1][1], 2)) + (4 * np.power(H[0][1], 2)))))
            Z[m][n] = np.abs(k1)
            print(np.floor(k1 - H[0][0]))
            theta[m][n] = Newton(np.floor(H[0][1]), np.floor(k1 - H[0][0]))
    return (Z, theta)

k1_curve, theta = Hesse(gxx, gyy, gxy)
# print(np.min(k1_curve))
# print(np.max(k1_curve))

cv2.imshow("XD3", theta)
cv2.waitKey()
cv2.destroyAllWindows()

# Funkcja Non-Maximum Suppression
# def non_maximum_suppression(G, theta):
#     M, N = G.shape
#     Z = np.zeros((M, N), dtype=np.float32)
    
#     for i in range(1, M-1):
#         for j in range(1, N-1):
#             angle = theta[i, j]

#             q = 255
#             r = 255

#             if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
#                 q = G[i, j+1]
#                 r = G[i, j-1]
#             elif 22.5 <= angle < 67.5:
#                 q = G[i+1, j-1]
#                 r = G[i-1, j+1]
#             elif 67.5 <= angle < 112.5:
#                 q = G[i+1, j]
#                 r = G[i-1, j]
#             elif 112.5 <= angle < 157.5:
#                 q = G[i-1, j-1]
#                 r = G[i+1, j+1]

#             if (G[i, j] >= q) and (G[i, j] >= r):
#                 Z[i, j] = G[i, j]
#             else:
#                 Z[i, j] = 0
#     return Z