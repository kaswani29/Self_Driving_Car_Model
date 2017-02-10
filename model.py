import csv
import cv2
import numpy as np
import pickle
import seaborn as sns


# Reading in Steering angle and path for image files
# Preprocessing can be done separately in another file but as most of the model run required tweaking dataset
# I chose to make it a part of this file itself

path = './data/'
lines = []
with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)


images = []
measurements = []
# This part builds the training set
# In order to improve the distribution images with steering angle very close to zero were removed
# Correction factor was introduced for images from left and right camera
for line in lines:
    measurement = abs(round(float(line[3]),4))
    # measurement = float(line[3])
    if measurement > 0.08:
        for i in range(3):
            source_path = line[0]
            tokens = source_path.split('/')
            filename = tokens[-1]
            local_path = path + "IMG/" + filename
            image = cv2.imread(local_path)
            images.append(image)
        correction = 0.2
        measurement = float(line[3])
        measurements.append(measurement)
        measurements.append(measurement+correction)
        measurements.append(measurement-correction)

# Getting new steering distribution
sns.distplot(measurements)
len(measurements)
len(images)
augmented_images = []
augmented_measurements = []

# Adding flipped images to reduce effect counterclock wise track, also increasing the data size
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image,1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
batch_size = 128  # The lower the better
nb_epoch = 10
# The higher the better

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D

# This flag is hard coded but can be made to pass as an argument if required
make_new_model = True

# This model is based on Nvidia deep learning architecture with some dropouts added to stop overfitting

if make_new_model:
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (1,1)),input_shape=(160, 320, 3) ))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(48, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # The optimizer used was 'adam' and Mean squared error was chosen as loss function
    model.compile(optimizer="adam", loss='mse')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=batch_size)
    model.save('model.h5')
else:
    # To resume training on an already saved model
    model = load_model('model.h5')
    model.compile(optimizer="adam", loss='mse')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, batch_size=batch_size)
    print ("Please input name for the new model")
    name = input()
    model.save(name)
