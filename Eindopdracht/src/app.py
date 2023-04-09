import cv2
from keras.models import load_model
from skimage import color
import matplotlib.pyplot as plt

#TODO change to relative paths (why not working?)
model = load_model("C:/Users/emmar/Documents/GitHub/VISN/Eindopdracht/src/ASL_model.keras")

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

live_capture_mode = False

#function to preprocess the image the same as the training data
def preprocess_image(image, image_size = 50):
    compressed_image = cv2.resize(image, (image_size, image_size))
    reshaped_image = compressed_image.reshape(-1, image_size, image_size, 1)
    normalized_image = (reshaped_image / 255) - 0.5
    return normalized_image

if live_capture_mode:
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
        preprocessed_image = preprocess_image(image)

        #predict the letter
        prediction = model.predict(preprocessed_image)

        #print the actual letter instead of percentages
        predicted_letter = list(prediction[0]).index(max(prediction[0]))
        print(prediction)
        print(letters[predicted_letter])
    
    #if capture went wrong
    else:
        print("No image was captured")


else:
    #for each letter
    for i in range(len(letters) - 3):
        #open the test image version of the letter
        path_captured = "C:/Users/emmar/Documents/GitHub/VISN/Eindopdracht/dataset/asl_alphabet_test/als_alphabet_test_captures/"+letters[i]+"_test.jpg"
        path = "C:/Users/emmar/Documents/GitHub/VISN/Eindopdracht/dataset/asl_alphabet_test/asl_alphabet_test/"+letters[i]+"_test.jpg"
        image = cv2.imread(path_captured, cv2.IMREAD_GRAYSCALE)

        #preprocess the image the same as the images used for training
        preprocessed_image = preprocess_image(image)

        #predict the letter
        prediction = model.predict(preprocessed_image)

        #print the actual letter instead of percentages
        predicted_letter = list(prediction[0]).index(max(prediction[0]))
        # print(prediction)
        print(letters[predicted_letter])