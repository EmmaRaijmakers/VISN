from skimage.viewer import ImageViewer
from skimage import io
from skimage.color import rgb2hsv
from matplotlib import pyplot as plt
 
img = io.imread("Opdracht_1/flower.jpg")
# io.imshow(img)

#TODO sneller door eerst de pixels aan te geven met 1 ipv 0 in matrix
#dan pixels aangegeven met 1 omzetten en dan samenvoegen (matrix vermenigvuldiging)

#function that converts image img to grayscale but keeps red pixels with a R-value
#within the threshold values min and max
def keep_red(img, thres_min, thres_max):
    for row in img:
        for pixel in row:
            if(not(pixel[0] >= thres_min and pixel[0] <= thres_max and pixel[0] > pixel[1] and pixel[0] > pixel[2])):
                grayscale_red = pixel[0] * 0.299
                grayscale_green = pixel[1] * 0.587
                grayscale_blue = pixel[2] * 0.114

                grayscale_value = int(grayscale_red + grayscale_green + grayscale_blue) 

                pixel[0] = grayscale_value
                pixel[1] = grayscale_value
                pixel[2] = grayscale_value


#function that creates a histogram of the hue values in the image img
def img_to_hue_histogram(img):
    hsv_img = rgb2hsv(img)
    hue_values = hsv_img[:, :, 0]
    print(hue_values)

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(hue_values)
    plt.show()

#histogram before grayscale
img_to_hue_histogram(img)

#show grayscale image
keep_red(img, 75, 255)
viewer = ImageViewer(img)
viewer.show()

#histogram after grayscale
img_to_hue_histogram(img)
