from skimage.viewer import ImageViewer
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import feature
from skimage.filters import gaussian

image = imread("hier_image_path")

#grayscale
image_gray = rgb2gray(image)

#edge detection
canny_filter = feature.canny(image, sigma=2.1) #TODO change sigma

#gaussian filter                    #for rgb image
gaussian_filter = gaussian(image, multichannel=True, sigma=2) #TODO change sigma

viewer = ImageViewer(image)
viewer.show()
