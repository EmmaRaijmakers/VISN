from skimage import data
from skimage.viewer import ImageViewer
from skimage import io
 
img = io.imread("Opdracht_1/flower.jpg")
# io.imshow(img)

viewer = ImageViewer(img)
viewer.show()