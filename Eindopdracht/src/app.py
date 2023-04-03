import cv2
from keras.models import load_model

#load model made in main file
#to not have to train the model everytime the application is run

#TODO change to relative paths (why not working?)
model = load_model("C:/Users/emmar/Documents/GitHub/VISN/Eindopdracht/src/ASL_model.keras")

webcam_port = 0
webcam = cv2.VideoCapture(webcam_port)
  
#reading the input using the webcam
result, image = webcam.read()
  
#if image is captured  
if result: 
    #show the capture
    cv2.imshow("ASL-detection", image)
  
    #destroy window after keyboard interrupt
    cv2.waitKey(0)
    cv2.destroyWindow("ASL-detection")


  
#if capture went wrong
else:
    print("No image was captured")

#load_model van checkpoint om model op te slaan en weer te loaden, zodat niet bij opstarten model te trainen