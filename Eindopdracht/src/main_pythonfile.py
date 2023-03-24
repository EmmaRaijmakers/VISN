#TODO remove libraries that are not used

from skimage.viewer import ImageViewer
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import feature
from skimage.filters import gaussian

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import random

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from keras.utils import to_categorical

#TODO write read me and give credit in read me to \/

#Dataset used for this project from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download

image = imread("C:/Users/emmar/Documents/GitHub/VISN/Opdracht_1/flower.jpg")

#grayscale
image_gray = rgb2gray(image)

#edge detection
canny_filter = feature.canny(image_gray, sigma=2) #TODO change sigma

#gaussian filter                    #for rgb image
gaussian_filter = gaussian(image, multichannel=True, sigma=2) #TODO change sigma

# viewer = ImageViewer(image)
# viewer.show()

# viewer = ImageViewer(image_gray)
# viewer.show()

# viewer = ImageViewer(canny_filter)
# viewer.show()

# viewer = ImageViewer(gaussian_filter)
# viewer.show()

dataset_dir = "Eindopdracht/dataset/asl_alphabet_train/asl_alphabet_train"
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

#tutorial: https://www.youtube.com/watch?v=j-3vuBynnOE

training_data = []

for letter in letters:
    #get directory of a certain letter
    path = os.path.join(dataset_dir, letter)

    #create number for each letter
    letter_num = letters.index(letter)
    print(letter)
    
    for image in os.listdir(path):
        #get one image
        image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE) #TODO gray scale can be added here, do it??

        #compress the image to a smaller resolution
        #TODO is this needed?? > makes faster
        image_size = 50 #TODO this bigger/smaller?
        compressed_image_array = cv2.resize(image_array, (image_size, image_size))

        #add canny filter
        # canny_filter_image = feature.canny(compressed_image_array, sigma=3)

        #add gaussian filter
        # gaussian_filter_image = gaussian(compressed_image_array, sigma=2) #TODO change sigma

        #add new image to the training set
        training_data.append([compressed_image_array, letter_num])

#randomize the images
random.shuffle(training_data)

train_images = []
train_labels = []

test_data_size = 10000

test_images = []
test_labels = []

#separate images and labels into the testing dataset
#(the first 10.000) and the training dataset (the rest)

#TODO can this be done easier? split?
for i in range(len(training_data)):
    if i < test_data_size:
        test_images.append(training_data[i][0])
        test_labels.append(training_data[i][1])
    else:
        train_images.append(training_data[i][0])
        train_labels.append(training_data[i][1])

#reshape train and test images
train_images = np.array(train_images).reshape(-1, image_size, image_size, 1)
test_images = np.array(test_images).reshape(-1, image_size, image_size, 1)

#normalize the images
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

#normalize the images canny
# train_images = train_images - 0.5
# test_images = test_images - 0.5

num_filters = 10
filter_size = 3
pool_size = 2
num_epochs = 50 #TODO change these vars

model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=train_images[0].shape),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(29, activation="sigmoid", name="dense")
])


model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, to_categorical(train_labels), epochs=num_epochs, validation_data=(test_images, to_categorical(test_labels)))

test_loss, test_acc = model.evaluate(test_images,  to_categorical(test_labels), verbose=2)
print(test_acc)