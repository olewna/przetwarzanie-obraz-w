from PIL import Image
import numpy as np

def okienko_sinusoidalne(image_path, save_file):
    image = Image.open(image_path).convert('L')
    img_array = np.array(image)
    M = img_array.shape[0]
    N = img_array.shape[1]

    m_indices, n_indices = np.mgrid[0:M, 0:N]
    w_mn = np.sin(np.pi * m_indices / M) * np.sin(np.pi * n_indices / N)

    result_image_array = img_array * w_mn

    result_image_array = np.clip(result_image_array, 0, 255).astype(np.uint8)

    result_image = Image.fromarray(result_image_array)
    result_image.save(save_file)
    result_image.show()

    return result_image

def filtr_sredni(image, save_file):
    img_array = np.array(image)
    M = img_array.shape[0]
    N = img_array.shape[1]

    output_array = np.zeros_like(img_array)

    for i in range(1, M-1):
        for j in range(1, N-1):
            region = img_array[i-1:i+2, j-1:j+2]
            
            region_sum = np.sum(region)
            
            output_array[i, j] = region_sum // 9

    output_image = Image.fromarray(output_array)

    output_image.save(save_file)

    output_image.show()

def zadanie_17():
    image_path = 'ptaki.png'
    sinus = okienko_sinusoidalne(image_path, "17a.png")
    filtr_sredni(sinus, "17b.png")
    image = Image.open(image_path).convert('L')
    filtr_sredni(image, "17d.png")

zadanie_17()