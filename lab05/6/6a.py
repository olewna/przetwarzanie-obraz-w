import cv2
import numpy as np

def isolated_point(image, threshold):
    # Przygotowanie macierzy na przefiltrowany obraz
    filtered_image = np.zeros_like(image, dtype=np.float32)
    
    # Wysokość i szerokość obrazu
    height, width = image.shape
    
    # Iteracja przez każdy piksel (pomijamy brzegi 1 piksela)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Pobranie sąsiadów piksela (włącznie z nim samym)
            neighbors = image[i - 1:i + 2, j - 1:j + 2].flatten()
            eight_neighbours = np.delete(neighbors, 4)

            mean = np.floor(eight_neighbours.sum() / 8)

            # if (i==200 and j==300):
            #     print(neighbors)
            #     print(eight_neighbours)
            #     print(mean)

            if (np.abs(image[i][j] - mean) < threshold):
                filtered_image[i][j] = image[i][j]
            else:
                filtered_image[i][j] = mean
    
    # Normalizacja i konwersja wyniku do formatu uint8
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image

# Przykład użycia:
input_image = cv2.imread('initial data/Jellyfish.png', cv2.IMREAD_GRAYSCALE)

# Zastosowanie eliminacji punktów izolowanych
filtered_image = isolated_point(input_image, 10)

output_path = '6/6a_result.png'
cv2.imwrite(output_path, filtered_image)
