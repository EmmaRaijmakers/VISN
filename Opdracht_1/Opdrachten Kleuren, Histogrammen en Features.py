from skimage import data
from skimage.viewer import ImageViewer
from skimage import io
 
img = io.imread("Opdracht_1/flower.jpg")
# io.imshow(img)

#TODO sneller door eerst de pixels aan te geven met 1 ipv 0 in matrix
#dan pixels aangegeven met 1 omzetten en dan samenvoegen (matrix vermenigvuldiging)

#function that converts image img to grayscale but keeps red pixels with a R-value
#greater than the threshold thres
def keep_red(img, thres):
    for row in img:
        print("next row")
        for pixel in row:
            if(not(pixel[0] > thres and pixel[0] > pixel[1] and pixel[0] > pixel[2])):
                grayscale_red = pixel[0] * 0.299
                grayscale_green = pixel[1] * 0.587
                grayscale_blue = pixel[2] * 0.114

                grayscale_value = int(grayscale_red + grayscale_green + grayscale_blue) 

                pixel[0] = grayscale_value
                pixel[1] = grayscale_value
                pixel[2] = grayscale_value

keep_red(img, 75)
            
viewer = ImageViewer(img)
viewer.show()