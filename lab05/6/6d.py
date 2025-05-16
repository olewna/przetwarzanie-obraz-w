import cv2
import numpy as np

def alphatrimmedmeanfilter_2d(image, alpha):
    M, N = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    
    # Start and end of the trimmed ordered set
    start = alpha // 2
    end = 9 - (alpha // 2)
    
    # Move window through all elements of the image
    for m in range(1, M - 1):
        for n in range(1, N - 1):
            # Pick up window elements
            window = image[m - 1:m + 2, n - 1:n + 2].flatten()
            
            # Order elements (only necessary part of them)
            for j in range(end):
                # Find position of minimum element
                min_idx = j
                for k in range(j + 1, 9):
                    if window[k] < window[min_idx]:
                        min_idx = k
                
                # Put found minimum element in its place
                window[j], window[min_idx] = window[min_idx], window[j]
            
            # Get result - the mean value of the elements in trimmed set
            result[m - 1, n - 1] = np.mean(window[start:end])
    
    return result.astype(np.uint8)

# Example usage:
image = cv2.imread('initial data/Jellyfish.png', cv2.IMREAD_GRAYSCALE)
alpha = 2
filtered_image = alphatrimmedmeanfilter_2d(image, alpha)

output_path = '6/6d_result.png'
cv2.imwrite(output_path, filtered_image)
