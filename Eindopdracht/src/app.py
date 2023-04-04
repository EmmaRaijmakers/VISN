import cv2
from keras.models import load_model
from skimage import color
import matplotlib.pyplot as plt

#load model made in main file
#to not have to train the model everytime the application is run

#TODO change to relative paths (why not working?)
model = load_model("C:/Users/emmar/Documents/GitHub/VISN/Eindopdracht/src/ASL_model.keras")

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

image_size = 50

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

    #preprocess the image the same as the images used for training
    gray_image = color.rgb2gray(image)
    compressed_image = cv2.resize(gray_image, (image_size, image_size))
    reshaped_image = compressed_image.reshape(-1, image_size, image_size, 1)
    normalized_image = (reshaped_image / 255) - 0.5

    #predict the letter
    prediction = model.predict(normalized_image)

    #print the actual letter instead of percentages
    predicted_letter = list(prediction[0]).index(max(prediction[0]))
    print(prediction)
    print(letters[predicted_letter])
  
#if capture went wrong
else:
    print("No image was captured")

#load_model van checkpoint om model op te slaan en weer te loaden, zodat niet bij opstarten model te trainen