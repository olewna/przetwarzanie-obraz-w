import cv2

def otsu_1(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    otsu_threshhold, new_image = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)      

    print(f"Próg: {otsu_threshhold}")
    cv2.imwrite('4a.png', new_image)

image1 = cv2.imread('roze.png') 
otsu_1(image1)
