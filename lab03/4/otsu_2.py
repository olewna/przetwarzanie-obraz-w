import numpy as np
from PIL import Image

def otsu_trojklasowe(image_path, save_file):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    
    pixel_values, counts = np.unique(img_array, return_counts=True)
    
    pixel_dict = dict(zip(pixel_values, counts))
    # print(pixel_dict)

    best_T = otsu(counts, pixel_values) # 176

    srednia1 = 0
    liczba_pikseli1 = 0
    # print(f"Optymalny prog Otsu: {best_T}")
    for i in range(best_T):
        if i in pixel_dict:
            srednia1 += i * pixel_dict[i]
            liczba_pikseli1 += pixel_dict[i]

    q_0 = int(np.floor(srednia1 / liczba_pikseli1)) # 107

    srednia2 = 0
    liczba_pikseli2 = 0
    for i in range(256):
        if i in pixel_dict and i >= best_T:
            srednia2 += i * pixel_dict[i]
            liczba_pikseli2 += pixel_dict[i]

    q_1 = int(np.floor(srednia2 / liczba_pikseli2)) # 245

    pixel_dict = binaryzacja(q_0, q_1, pixel_dict)
   
    # print(pixel_dict)

    pixel_values = np.array(list(pixel_dict.keys()))
    counts = np.array(list(pixel_dict.values()))

    best_T = otsu(counts, pixel_values) # 129
    # print(best_T) 

    new_img = np.where(img_array < best_T, 0, 255).astype(np.uint8)
    new_img_pil = Image.fromarray(new_img)
    
    new_img_pil.save(save_file)

    
def otsu(counts, pixel_values):
    total_pixels = np.sum(counts)
    probabilities = counts / total_pixels

    best_T = 0
    max_variance = 0

    mean_global = np.sum(pixel_values * probabilities)

    w1 = 0
    sum1 = 0 

    for T in pixel_values[:-1]:
        idx = np.where(pixel_values == T)[0][0]
        w1 += probabilities[idx]
        sum1 += T * probabilities[idx]
        
        if w1 == 0 or w1 == 1:
            continue
        
        w2 = 1 - w1
        mean1 = sum1 / w1
        mean2 = (mean_global - sum1) / w2
        
        variance_between = w1 * w2 * (mean1 - mean2) ** 2
        
        if variance_between > max_variance:
            max_variance = variance_between
            best_T = T

    return best_T

def binaryzacja(q_0, q_1, pixel_dict):
    pixele_czarne = 0
    pixele_biale = 0

    for i in range(q_0+1):
        if i in pixel_dict:
            pixele_czarne += pixel_dict[i]

    for i in range(256):
        if i in pixel_dict and i >= q_1:
            pixele_biale += pixel_dict[i]

    # print(pixel_dict)
    # print(f"{q_0}: {pixele_czarne}")
    # print(f"{q_1}: {pixele_biale}")

    pixel_dict = {k: v for k, v in pixel_dict.items() if k > q_0}
    pixel_dict = {k: v for k, v in pixel_dict.items() if k < q_1}
    pixel_dict[0] = pixele_czarne
    pixel_dict[255] = pixele_biale
    pixel_dict = dict(sorted(pixel_dict.items()))

    return pixel_dict

image_path = "roze.png"
otsu_trojklasowe(image_path, "4b.png")