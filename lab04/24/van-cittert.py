import cv2
import numpy as np
import sys
from scipy.signal import convolve2d

def van_cittert_deconvolution(y, h, num_iterations_list):
    """
    Wykonuje dekonwolucję obrazu algorytmem Van-Citterta:
      g^(k+1) = g^(k) + [y - h*g^(k)]
    dla k w zakresie 0..max(num_iterations_list),
    a następnie zwraca wyniki dla żądanych iteracji z listy num_iterations_list.
    
    Parametry:
    -----------
    y : np.ndarray
        Obraz wejściowy (np. 2D, skala szarości), typ float32/float64 lub uint8.
    h : np.ndarray
        Jądro filtra (2D), również w typie float.
    num_iterations_list : list
        Lista wybranych liczb iteracji (np. [2, 5, 15]), 
        dla których chcemy zwrócić wynik.
    
    Zwraca:
    --------
    results : dict
        Słownik postaci { k : (g_k, diff_k) }, gdzie:
          - g_k to obraz po k iteracjach,
          - diff_k to (g_k - y).
    """
    # Konwersja obrazu wejściowego do float (dla bezpieczeństwa obliczeń)
    y_float = y.astype(np.float32)
    
    # Inicjalizacja: g^(0) = y
    g_current = y_float.copy()
    
    # Wyznaczamy maksymalną liczbę iteracji, jaką musimy wykonać
    max_iter = max(num_iterations_list)
    
    # Przygotowujemy słownik na wyniki
    results = {}
    
    # Iteracyjny algorytm Van-Citterta
    for k in range(1, max_iter + 1):
        # Obliczamy konwolucję h*g^(k)
        # Uwaga: "mode='same', boundary='symm'" -> odbicie brzegów (symetryczne)
        hg = convolve2d(g_current, h, mode='same', boundary='symm')
        
        # Aktualizacja: g^(k+1) = g^(k) + [y - h*g^(k)]
        g_next = g_current + (y_float - hg)
        
        # Przejście do kolejnej iteracji
        g_current = g_next
        
        # Jeśli k jest na liście żądanych iteracji, zapisz wynik
        if k in num_iterations_list:
            # Obraz wynikowy można przyciąć i/lub zrzutować do uint8 
            # w razie potrzeby wizualizacji.
            g_k = np.clip(g_current, 0, 255).astype(np.uint8)
            
            # Różnica (g_k - y) – też w formie do wizualizacji:
            diff_k = (g_current - y_float)  # różnica w float
            diff_k_vis = np.clip(diff_k + 128, 0, 255).astype(np.uint8)
            # powyżej: przesunięcie o +128, by wartości ujemne były widoczne 
            # w zakresie 0..255 (to często stosowana metoda poglądowa).
            
            results[k] = (g_k, diff_k_vis)
    
    return results

def main():    
    input_image_path = "bocian_filtered.png"
    y = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Definicja filtra h (np. 5x5 - współczynniki dwumianu Newtona)
    # Przykładowa macierz:
    #   1   4   6   4   1
    #   4  16  24  16   4
    #   6  24  36  24   6
    #   4  16  24  16   4
    #   1   4   6   4   1
    # Suma = 256
    h = np.array([
        [1,  4,  6,  4,  1],
        [4, 16, 24, 16,  4],
        [6, 24, 36, 24,  6],
        [4, 16, 24, 16,  4],
        [1,  4,  6,  4,  1]
    ], dtype=np.float32) / 256.0
    
    # Lista wybranych iteracji, np. k = 2, 5, 15
    iterations = [2, 5, 15]
    
    # Uruchamiamy algorytm Van-Citterta
    results = van_cittert_deconvolution(y, h, iterations)
    
    # Zapisujemy wyniki
    for k in iterations:
        g_k, diff_k = results[k]
        
        # Nazwy plików wynikowych
        out_g_path = f"24_result_{k}.png"
        out_diff_path = f"24_diff_{k}.png"
        
        cv2.imwrite(out_g_path, g_k)
        cv2.imwrite(out_diff_path, diff_k)
        
        print(f"Zapisano wynik dekonwolucji po {k} iteracjach do: {out_g_path}")
        print(f"Zapisano obraz różnicy (g^{k} - y) do: {out_diff_path}")

if __name__ == "__main__":
    main()