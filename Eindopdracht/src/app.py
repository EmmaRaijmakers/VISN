from cv2 import *

webcam_port = 0
webcam = VideoCapture(webcam_port)
  
#reading the input using the webcam
result, image = webcam.read()
  
#if image is captured  
if result: 
    #show the capture
    imshow(image)
  
    #destroy window after keyboard interrupt
    waitKey(0)
    destroyWindow("GeeksForGeeks")
  
#if capture went wrong
else:
    print("No image was captured")