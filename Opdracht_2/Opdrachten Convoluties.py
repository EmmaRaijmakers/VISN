from skimage import data, filters
from skimage.viewer import ImageViewer
import scipy
from scipy import ndimage

image = data.camera()
viewer = ImageViewer(image)
viewer.show()

mask1=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]

#Opdracht 1
mask_edgedetection1 = [[-1,0,1],[-1,0,1],[-1,0,1]]
mask_edgedetection2 = [[-1,-1,-1],[0,0,0],[1,1,1]]
mask_edgedetection3 = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]

# newimage=scipy.ndimage.convolve(image, mask1)
# newimage=scipy.ndimage.convolve(newimage, mask1)

newimage1 = scipy.ndimage.convolve(image, mask_edgedetection1)
viewer = ImageViewer(newimage1)
viewer.show()

newimage2 = scipy.ndimage.convolve(image, mask_edgedetection2)
viewer = ImageViewer(newimage2)
viewer.show()

newimage3 = scipy.ndimage.convolve(image, mask_edgedetection3)
viewer = ImageViewer(newimage3)
viewer.show()

#Opdracht 2

#Wat opvalt aan onderstaande filters in vergelijking met bovenstaande,
#is dat onderstaande filters veel beter werken. 

#Prewitt
prewitt_filter = filters.prewitt(image)
viewer = ImageViewer(prewitt_filter)
viewer.show()

#Scharr
scharr_filter = filters.scharr(image)
viewer = ImageViewer(scharr_filter)
viewer.show()

#Sobel
sobel_filter = filters.sobel(image)
viewer = ImageViewer(sobel_filter)
viewer.show()

#Opdracht 3