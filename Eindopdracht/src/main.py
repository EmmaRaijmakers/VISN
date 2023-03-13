from skimage.io import imread
from skimage.color import rgb2gray

image = imread("hier_image_path")

#grayscale
image_gray = rgb2gray(image)
