import numpy as np
from PIL import Image

def apply_dithering(image):
    # Macierz ditheringu 4x4
    dither_matrix = np.array([
        [6, 14, 2, 8],
        [4, 0, 10, 11],
        [12, 15, 5, 1],
        [9, 3, 13, 7]
    ])

    # Normalizowanie macierzy do zakresu 0-1
    dither_matrix = dither_matrix / 16.0 * 255

    # Wczytanie obrazu
    img = image.convert('L')  # Konwertujemy na obraz w odcieniach szarości
    pixels = np.array(img)
    
    # Wymiary obrazu
    height, width = pixels.shape
    
    # Zakres poziomów szarości
    num_gray_levels = 256  # Zakładając 256 poziomów szarości

    # Przygotowanie nowego obrazu
    new_height = height * 4
    new_width = width * 4
    new_pixels = np.ones((new_height, new_width), dtype=np.uint8) * 255  # Tło białe

    # Proces ditheringu
    for y in range(height):
        for x in range(width):
            pixel_value = pixels[y, x]
            # Zmienna do śledzenia liczby większych wartości w macierzy
            count_black = 0

            # Sprawdzanie wartości w macierzy 4x4
            for i in range(4):
                for j in range(4):
                    if dither_matrix[i, j] > pixel_value:
                        count_black += 1

            # Zmiana piksela na blok 4x4
            for i in range(4):
                for j in range(4):
                    if dither_matrix[i, j] > pixel_value:
                        new_pixels[4*y + i, 4*x + j] = 0  # Piksel czarny
                    else:
                        new_pixels[4*y + i, 4*x + j] = 255  # Piksel biały

    # Zapisz wynikowy obraz
    dithered_image = Image.fromarray(new_pixels)
    return dithered_image

if __name__ == "__main__":
    import os
    image_path = os.path.join(os.getcwd(), 'rejtan.png')
    input_image = Image.open(image_path)
    new_image = apply_dithering(input_image)

    new_image.save('rejtan_3.png')