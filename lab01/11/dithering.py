import numpy as np
from PIL import Image

def floyd_steinberg_dithering(image, threshold=128):
    # Convert image to grayscale
    grayscale_image = image.convert('L')
    
    # Convert image to numpy array
    img_array = np.array(grayscale_image)
    
    # Floyd-Steinberg dithering algorithm
    height, width = img_array.shape
    for y in range(height):
        for x in range(width):
            old_pixel = img_array[y, x]
            new_pixel = 0 if old_pixel < threshold else 255
            img_array[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            # Distribute the error to neighboring pixels
            if x + 1 < width:
                img_array[y, x + 1] += quant_error * 7 // 16
            if y + 1 < height:
                if x - 1 >= 0:
                    img_array[y + 1, x - 1] += quant_error * 3 // 16
                img_array[y + 1, x] += quant_error * 5 // 16
                if x + 1 < width:
                    img_array[y + 1, x + 1] += quant_error * 1 // 16
    
    # Convert numpy array back to image
    dithered_image = Image.fromarray(img_array)
    
    return dithered_image

def floyd_steinberg_dithering_5_levels(image):
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image, dtype=np.int32)
    
    levels = [0, 64, 128, 192, 255]
    height, width = img_array.shape
    
    for y in range(height):
        for x in range(width):
            old_pixel = img_array[y, x]
            new_pixel = min(levels, key=lambda v: abs(v - old_pixel))
            img_array[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            if x + 1 < width:
                img_array[y, x + 1] += quant_error * 7 // 16
            if y + 1 < height:
                if x - 1 >= 0:
                    img_array[y + 1, x - 1] += quant_error * 3 // 16
                img_array[y + 1, x] += quant_error * 5 // 16
                if x + 1 < width:
                    img_array[y + 1, x + 1] += quant_error * 1 // 16
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def floyd_steinberg_dithering(image, threshold=128):
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image, dtype=np.int32)
    
    height, width = img_array.shape
    for y in range(height):
        for x in range(width):
            old_pixel = img_array[y, x]
            new_pixel = 0 if old_pixel < threshold else 255
            img_array[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            if x + 1 < width:
                img_array[y, x + 1] += quant_error * 7 // 16
            if y + 1 < height:
                if x - 1 >= 0:
                    img_array[y + 1, x - 1] += quant_error * 3 // 16
                img_array[y + 1, x] += quant_error * 5 // 16
                if x + 1 < width:
                    img_array[y + 1, x + 1] += quant_error * 1 // 16
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def floyd_steinberg_dithering_5_levels(image):
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image, dtype=np.int32)
    
    levels = [0, 64, 128, 192, 255]
    height, width = img_array.shape
    
    for y in range(height):
        for x in range(width):
            old_pixel = img_array[y, x]
            new_pixel = min(levels, key=lambda v: abs(v - old_pixel))
            img_array[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            if x + 1 < width:
                img_array[y, x + 1] += quant_error * 7 // 16
            if y + 1 < height:
                if x - 1 >= 0:
                    img_array[y + 1, x - 1] += quant_error * 3 // 16
                img_array[y + 1, x] += quant_error * 5 // 16
                if x + 1 < width:
                    img_array[y + 1, x + 1] += quant_error * 1 // 16
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def jarvis_judice_ninke_dithering(image, threshold=128):
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image, dtype=np.int32)
    
    height, width = img_array.shape
    diffusion_matrix = [(1, 0, 7/48), (2, 0, 5/48), (-2, 1, 3/48), (-1, 1, 5/48),
                        (0, 1, 7/48), (1, 1, 5/48), (2, 1, 3/48), (-2, 2, 1/48),
                        (-1, 2, 3/48), (0, 2, 5/48), (1, 2, 3/48), (2, 2, 1/48)]
    
    for y in range(height):
        for x in range(width):
            old_pixel = img_array[y, x]
            new_pixel = 0 if old_pixel < threshold else 255
            img_array[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            for dx, dy, factor in diffusion_matrix:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    img_array[ny, nx] += quant_error * factor
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def jarvis_judice_ninke_dithering_5_levels(image):
    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image, dtype=np.int32)
    
    levels = [0, 64, 128, 192, 255]
    height, width = img_array.shape
    diffusion_matrix = [(1, 0, 7/48), (2, 0, 5/48), (-2, 1, 3/48), (-1, 1, 5/48),
                        (0, 1, 7/48), (1, 1, 5/48), (2, 1, 3/48), (-2, 2, 1/48),
                        (-1, 2, 3/48), (0, 2, 5/48), (1, 2, 3/48), (2, 2, 1/48)]
    
    for y in range(height):
        for x in range(width):
            old_pixel = img_array[y, x]
            new_pixel = min(levels, key=lambda v: abs(v - old_pixel))
            img_array[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            for dx, dy, factor in diffusion_matrix:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    img_array[ny, nx] += quant_error * factor
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))


# Example usage:
# if __name__ == "__main__":
input_image_path = 'rejtan.png'
output_image_fs = 'rejtan_1a.png'
output_image_fs_5 = 'rejtan_1b.png'
output_image_jjn = 'rejtan_2a.png'
output_image_jjn_5 = 'rejtan_2b.png'

input_image = Image.open(input_image_path)

dithered_image = floyd_steinberg_dithering(input_image, threshold=128)
dithered_image.save(output_image_fs)

dithered_image_5 = floyd_steinberg_dithering_5_levels(input_image)
dithered_image_5.save(output_image_fs_5)

dithered_image_jjn = jarvis_judice_ninke_dithering(input_image, threshold=128)
dithered_image_jjn.save(output_image_jjn)

dithered_image_jjn_5 = jarvis_judice_ninke_dithering_5_levels(input_image)
dithered_image_jjn_5.save(output_image_jjn_5)