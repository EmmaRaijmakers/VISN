from skimage import data, filters
from skimage.viewer import ImageViewer
import scipy
from scipy import ndimage

image = data.camera()
viewer = ImageViewer(image)
viewer.show()

mask1=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]

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