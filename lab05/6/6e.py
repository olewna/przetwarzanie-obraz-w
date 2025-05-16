import cv2
import numpy as np

def k_nearest_neighbor_filter(image, k=6):
    # Przygotowanie macierzy na przefiltrowany obraz
    filtered_image = np.zeros_like(image, dtype=np.float32)
    
    # Wysokość i szerokość obrazu
    height, width = image.shape
    
    # Iteracja przez każdy piksel (pomijamy brzegi 1 piksela)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Pobranie sąsiadów piksela (włącznie z nim samym)
            neighbors = image[i - 1:i + 2, j - 1:j + 2].flatten()
            
            # Sortowanie wartości sąsiadów
            sorted_neighbors = np.sort(neighbors)
            
            # Obliczenie średniej ważonej z użyciem k najbliższych sąsiadów
            filtered_image[i, j] = np.mean(sorted_neighbors[:k])
    
    # Normalizacja i konwersja wyniku do formatu uint8
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image

# Przykład użycia:
input_image = cv2.imread('initial data/Jellyfish.png', cv2.IMREAD_GRAYSCALE)

# Zastosowanie filtru k-Nearest Neighbor
filtered_image = k_nearest_neighbor_filter(input_image, k=6)

output_path = '6/6e_result.png'
cv2.imwrite(output_path, filtered_image)
