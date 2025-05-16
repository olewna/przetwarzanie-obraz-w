import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def get_pixel_value_dict(image_path):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    
    pixel_values, counts = np.unique(img_array, return_counts=True)
    
    pixel_dict = dict(zip(pixel_values, counts))

    # for i in range(256):
    #     if i not in pixel_dict:
    #         pixel_dict[i] = 0
        
    total_pixels = img_array.shape[0] * img_array.shape[1]

    cumulative_sum = 0
    for i in range(256):
        if i in pixel_dict:
            cumulative_sum += pixel_dict[i]
            pixel_dict[i] = (1 / total_pixels) * cumulative_sum
    
    return pixel_dict

def plot_histogram_from_dict(pixel_dict, save_name):
    values = list(pixel_dict.keys())
    counts = list(pixel_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(values, counts, width=1, color='black', alpha=0.7)

    plt.title("Skumulowany histogram")
    plt.xlabel("Wartość piksela")
    plt.ylabel("Skumulowana liczba pikseli")
    plt.grid(True)
    plt.savefig(save_name)
    plt.close()

def zadanie_7a():
    image_path = "czaszka.png"
    pixel_dict = get_pixel_value_dict(image_path)
    plot_histogram_from_dict(pixel_dict, "7a.png")

# zadanie_7a()

def wyrownanie(image_path):
    histogram = get_pixel_value_dict(image_path)
    G = len(histogram)
    H_equal = {}

    for i in range(256):
        if i in histogram:
            H_equal[i] = math.floor((G-1) * histogram[i])

    return H_equal

def plot_7b(pixel_dict, save_name):
    values = list(pixel_dict.keys())
    counts = list(pixel_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(values, counts, width=1, color='black', alpha=0.7)

    plt.title("Wyrownany histogram")
    plt.xlabel("Wartość piksela")
    plt.ylabel("Liczba pikseli")
    plt.grid(True)
    plt.savefig(save_name)
    plt.close()

def apply_histogram_equalization(image_path, H_equal, save_name):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    
    new_img_array = np.copy(img_array)
    
    for i in range(256):
        if i in H_equal:
            new_img_array[img_array == i] = H_equal[i]
    
    new_img = Image.fromarray(new_img_array)
    
    new_img.save(save_name)
    new_img.show()

def zadanie_7b():
    image_path = "czaszka.png"
    wyr = wyrownanie(image_path)
    plot_7b(wyr, "7b.png")
    apply_histogram_equalization(image_path, wyr, "7b_obraz.png")

# zadanie_7b()

def hiperbolizacja(image_path):
    histogram = get_pixel_value_dict(image_path)
    G = len(histogram)
    alfa = -1/3
    indeks = 1/(alfa+1)
    H_hyper = {}

    for i in range(256):
        if i in histogram:
            H_hyper[i] = math.floor((G-1) * math.pow(histogram[i], indeks))

    return H_hyper

def plot_7c(pixel_dict, save_name):
    values = list(pixel_dict.keys())
    counts = list(pixel_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(values, counts, width=1, color='black', alpha=0.7)

    plt.title("Wyrownany histogram")
    plt.xlabel("Wartość piksela")
    plt.ylabel("Liczba pikseli")
    plt.grid(True)
    plt.savefig(save_name)
    plt.close()

def apply_histogram_hyperbolization(image_path, H_hyper, save_name):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    
    new_img_array = np.copy(img_array)
    
    for i in range(256):
        if i in H_hyper:
            new_img_array[img_array == i] = H_hyper[i]
    
    new_img = Image.fromarray(new_img_array)
    
    new_img.save(save_name)
    new_img.show()

def zadanie_7c():
    image_path = "czaszka.png"
    hiper = hiperbolizacja(image_path)
    print(hiper)
    plot_7c(hiper, "7c.png")
    apply_histogram_hyperbolization(image_path, hiper, "7c_obraz.png")

# zadanie_7c()

def zadanie10():
    image_path = "CalunTurynskiPoKontrascie.jpg"
    wyr = wyrownanie(image_path)
    plot_7b(wyr, "10_v1.png")
    apply_histogram_equalization(image_path, wyr, "10_v1_obraz.png")

    hiper = hiperbolizacja(image_path)
    plot_7c(hiper, "10_v2.png")
    apply_histogram_hyperbolization(image_path, hiper, "10_v2_obraz.png")

zadanie10()