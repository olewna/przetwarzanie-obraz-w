from PIL import Image
import numpy as np

def czerwony_dla_szarego(value):
    if value <= 31 or value >= 223:
        return 0
    elif 31 < value <= 95:
        return int((value - 31) * (255 / (95 - 31)))
    elif 95 < value <= 159:
        return 255
    elif 159 < value <= 223:
        return int(255 - (value - 159) * (255 / (223 - 159)))  
    return 0

def zielony_dla_szarego(value):
    if value <= 95:
        return 0
    elif 95 < value <= 159:
        return int((value - 95) * (255 / (159 - 95)))
    elif 159 < value <= 223:
        return 255
    elif 223 < value:
        return int((-127 / 32) * (value - 223) + 255)
    return 0

def niebieski_dla_szarego(value):
    if 0 <= value <= 31:
        return int((127 / 31) * value + 128)
    elif 31 < value <= 95:
        return 255
    elif 95 < value <= 159:
        return int((-255 / 64) * (value - 95) + 255) 
    else:
        return 0

def zadanie_9():
    image_path = 'jezus.jpg'
    image = Image.open(image_path).convert('L')

    image_array = np.array(image)

    image_color = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            gray_value = image_array[i, j]
            red = czerwony_dla_szarego(gray_value)
            blue = niebieski_dla_szarego(gray_value)
            green = zielony_dla_szarego(gray_value)
            image_color[i, j] = [red, green, blue]

    color_image = Image.fromarray(image_color)

    color_image.save('10.png')

zadanie_9()