from PIL import Image
import numpy as np

def pad_with_reflection(image_array, pad_size):
    """ Odbicie lustrzane dla pikseli na krawędziach obrazu. """
    return np.pad(image_array, pad_size, mode='reflect')

def otsu_threshold(window):
    """ Oblicza próg Otsu dla danego okna obrazu. """
    hist, _ = np.histogram(window, bins=256, range=(0, 255))
    total_pixels = window.size
    best_threshold = 0
    max_variance = 0
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0
    weight_background = 0
    weight_foreground = 0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance_between > max_variance:
            max_variance = variance_between
            best_threshold = t
    return best_threshold

def local_otsu_thresholding(image_array, window_size=11):
    """ Stosuje lokalne progowanie Otsu dla każdego piksela obrazu. """
    pad_size = window_size // 2
    padded_image = pad_with_reflection(image_array, pad_size)
    binary_result = np.zeros_like(image_array, dtype=np.uint8)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            window = padded_image[i:i + window_size, j:j + window_size]
            threshold = otsu_threshold(window)
            binary_result[i, j] = 255 if image_array[i, j] > threshold else 0

    return binary_result

# 1. Wczytaj obraz
image = Image.open("roze.png").convert("L")  # Konwersja do skali szarości
image_array = np.array(image)

# 2. Zastosuj lokalne progowanie metodą Otsu
binary_image = local_otsu_thresholding(image_array)

# 3. Zapisz wynikowy obraz
result_image = Image.fromarray(binary_image)
result_image.save("4c.png")
print("Obraz wynikowy zapisano jako 4c.png")
