import numpy as np
import mahotas as mh
import matplotlib.pyplot as mtplt

img = mh.imread('21/21_2.bmp')

data = np.asarray(img)
input_image = np.where(data == 255, 0, np.where(data == 0, 255, data))
input_image2 = np.where(data == 255, 0, np.where(data == 0, 1, data))

kernel = np.array((
    [0, 0, 0],
    [1, 1, 1],
    [2, 1, 2],
    [2, 1, 2],
    [2, 1, 2],
    [0, 1, 0],
    [2, 1, 2],
    [2, 1, 2],
    [2, 1, 2],
    [1, 1, 1],
    [2, 1, 2],
    [2, 1, 2],
    [2, 1, 2],
    [0, 1, 0],
    [2, 1, 2],
    [2, 1, 2],
    [2, 1, 2],
    [1, 1, 1],
    [2, 1, 2],
    [2, 1, 2],
    [2, 1, 2],
    [0, 1, 0],
    [2, 1, 2],
    [2, 1, 2],
    [2, 1, 2],
    [1, 1, 1],
    [2, 1, 2],
    [2, 1, 2],
    [2, 1, 2],
    [0, 1, 0],
    [2, 1, 2],
    [2, 1, 2],
    [2, 1, 2],
    [1, 1, 1],
    [0, 0, 0]), dtype="int")
kernel_test = np.array([[0,0,0],[0,1,1],[0,0,0]])
kernel2 = np.array([[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[0]])
kernel3 = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])

output_image = mh.hitmiss(input_image2, kernel2)

fig, axes = mtplt.subplots(1, 2)
axes[0].imshow(input_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].set_axis_off()
axes[1].imshow(output_image, cmap='gray')
axes[1].set_title('Hit & Miss Transformed Image')
axes[1].set_axis_off()
mtplt.tight_layout()
mtplt.show()