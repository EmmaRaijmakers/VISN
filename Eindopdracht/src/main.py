from skimage.io import imread
from skimage.color import rgb2gray
from skimage import feature

image = imread("hier_image_path")

#grayscale
image_gray = rgb2gray(image)

#edge detection
canny_filter = feature.canny(image, sigma=2.1) #TODO change sigma
