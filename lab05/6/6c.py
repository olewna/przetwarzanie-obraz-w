from PIL import Image
import numpy as np

def mid_range_filter():
    image_path = 'initial data/Jellyfish.png'
    image = Image.open(image_path).convert('L')

    image_array = np.array(image)

    image_filtered = np.zeros_like(image_array, dtype=np.uint8)

    height, width = image_array.shape

    c = 0
    for i in range(0, height):
        for j in range(0, width):
            if (j == 959 and i == 761):
                print(f"Koniec")
                neighbors = [
                    image_array[abs(i - 1), abs(j - 1)], image_array[abs(i - 1), abs(j)], image_array[abs(i - 1), abs(j - 1)],
                    image_array[abs(i), abs(j - 1)],     image_array[abs(i), abs(j)],     image_array[abs(i), abs(j - 1)],
                    image_array[abs(i - 1), abs(j - 1)], image_array[abs(i - 1), abs(j)], image_array[abs(i - 1), abs(j - 1)]
                ]
            elif (j == 959):
                # print(f"j = {j}")
                neighbors = [
                    image_array[abs(i - 1), abs(j - 1)], image_array[abs(i - 1), abs(j)], image_array[abs(i - 1), abs(j - 1)],
                    image_array[abs(i), abs(j - 1)],     image_array[abs(i), abs(j)],     image_array[abs(i), abs(j - 1)],
                    image_array[abs(i + 1), abs(j - 1)], image_array[abs(i + 1), abs(j)], image_array[abs(i + 1), abs(j - 1)]
                ]
            elif (i == 761):
                # print(f"i = {i}")
                neighbors = [
                    image_array[abs(i - 1), abs(j - 1)], image_array[abs(i - 1), abs(j)], image_array[abs(i - 1), abs(j + 1)],
                    image_array[abs(i), abs(j - 1)],     image_array[abs(i), abs(j)],     image_array[abs(i), abs(j + 1)],
                    image_array[abs(i - 1), abs(j - 1)], image_array[abs(i - 1), abs(j)], image_array[abs(i - 1), abs(j + 1)]
                ]
            else:
                neighbors = [
                    image_array[abs(i - 1), abs(j - 1)], image_array[abs(i - 1), abs(j)], image_array[abs(i - 1), abs(j + 1)],
                    image_array[abs(i), abs(j - 1)],     image_array[abs(i), abs(j)],     image_array[abs(i), abs(j + 1)],
                    image_array[abs(i + 1), abs(j - 1)], image_array[abs(i + 1), abs(j)], image_array[abs(i + 1), abs(j + 1)]
                ]
            min_neighbor = np.min(neighbors)
            max_neighbor = np.max(neighbors)
            new_value = (np.add(max_neighbor,min_neighbor, dtype=np.int16)//2)

            # if (image_array[i,j] ==0 and c <= 3):
            # if (j == 959 and i == 761):
            #     print(F"NEIGHBORS: {neighbors}")
            #     print(f"MIN: {min_neighbor}")
            #     print(F"MAX: {max_neighbor}")
            #     print(f"NEW VALUE: {new_value}")
                # c += 1
                
            # print(f"j = {j}")
            image_filtered[i, j] = new_value

    filtered_image = Image.fromarray(image_filtered)

    filtered_image.save('6/6c_result.png')

mid_range_filter()