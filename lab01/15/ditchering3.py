from PIL import Image
import numpy as np

# Macierz Bayera 8x8 (Twoja dostarczona macierz)
bayer_matrix = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]
])

# Normalizacja macierzy Bayera do zakresu 0-255
bayer_matrix = bayer_matrix / 64.0  # 63 jest maksymalną wartością w macierzy

# Funkcja do ditheringu z wykorzystaniem macierzy Bayera
def apply_dithering_bayer(image_path, grayscale_palette=False):
    # Wczytanie obrazu
    img = Image.open(image_path).convert('L')  # Konwertujemy na obraz w odcieniach szarości
    pixels = np.array(img)
    
    # Wymiary obrazu
    height, width = pixels.shape
    
    # Tworzenie nowego obrazu
    new_pixels = np.zeros_like(pixels)

    # Zastosowanie ditheringu z macierzą Bayera
    if (not grayscale_palette):
        for y in range(height):
            for x in range(width):
                pixel_value = pixels[y, x]

                value = (pixel_value + 256 * (bayer_matrix[y % 8,x % 8] - 0.5))
                if value >=128:
                    new_pixels[y, x] = 255
                else:
                    new_pixels[y, x] = 0
    else:
        for y in range(height):
            for x in range(width):
                pixel_value = pixels[y, x]

                value = (pixel_value + 256 * (bayer_matrix[y % 8,x % 8] - 0.5))
                if (value < 64):
                    new_pixels[y, x] = 50
                elif (value < 128):
                    new_pixels[y, x] = 100
                elif (value < 192):
                    new_pixels[y, x] = 150
                else:
                    new_pixels[y, x] = 200

    # Zapisz obraz po ditheringu
    dithered_image = Image.fromarray(new_pixels)
    return dithered_image

# Ścieżka do obrazu
image_path = 'rejtan.png'

# Dithering z 1-bitową paletą (czarno-biała)
image = apply_dithering_bayer(image_path)
image.save('rejtan_4a.jpg')

# Dithering z paletą 4 wartości {50, 100, 150, 200}
image2 = apply_dithering_bayer(image_path, True)
image2.save('rejtan_4b.jpg')
