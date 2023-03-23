import cv2 as cv

webcam_port = 0
webcam = cv.VideoCapture(webcam_port)
  
#reading the input using the webcam
result, image = webcam.read()
  
#if image is captured  
if result: 
    #show the capture
    cv.imshow("ASL-detection", image)
  
    #destroy window after keyboard interrupt
    cv.waitKey(0)
    cv.destroyWindow("ASL-detection")
  
#if capture went wrong
else:
    print("No image was captured")