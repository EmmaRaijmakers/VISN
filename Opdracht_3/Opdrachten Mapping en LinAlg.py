from skimage import data
from skimage import transform
from skimage.viewer import ImageViewer
import numpy as np

#function to show an image before and after a transformation matrix is applied
def before_and_after_transform(img, scl = None, rot = None, trans = None):
    
    #show original image
    viewer = ImageViewer(img)
    viewer.show()

    #create new image using matrix
    transform_matrix = transform.AffineTransform(scale=scl, rotation=rot, translation=trans)
    new_img = transform.warp(img, transform_matrix)

    #show new image
    viewer = ImageViewer(new_img)
    viewer.show()

image = data.camera()

before_and_after_transform(image, scl=(2, 2)) #half the size in the x and y direction
before_and_after_transform(image, rot=np.pi/4) #turn 1/4 pi rad
before_and_after_transform(image, trans=(50, -100)) #move 50 pixels in the x direction and 100 in the y
before_and_after_transform(image, scl=(2, 2), rot=np.pi/4, trans=(50, -100)) #all of it