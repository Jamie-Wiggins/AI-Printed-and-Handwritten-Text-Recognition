import os
import time
import cv2
import numpy as np
from imageio import imread, imsave
from skimage.feature import hog
from sklearn.svm import LinearSVC
from skimage.morphology import label
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from emnist import extract_training_samples
from predictor import prediction

# load data from emnist
image, labels = extract_training_samples('letters')
# split data 70/30
data_slice = 70000
data_slice2 = 100000
# load training data
train_images = image[:data_slice,:,:]
train_labels = labels[:data_slice,]
# load test data
test_images = image[data_slice:data_slice2,:,:]
test_labels = labels[data_slice:data_slice2,]

train_data = []
# loop through train data
for image in train_images:
    hog_features = hog(image, orientations=12, pixels_per_cell=(2, 2), cells_per_block=(1, 1))
    # append hog features to new training data array
    train_data.append(hog_features)

test_data = []
# loop through testing data
for image in test_images:
    hog_features = hog(image, orientations=12, pixels_per_cell=(2, 2), cells_per_block=(1, 1))
    # append hog features to new testing data array
    test_data.append(hog_features)

# create SVC model
clf = LinearSVC(dual=True, verbose=0)
# train the model
clf.fit(train_data, train_labels)
# print out the result to console
print(clf.score(test_data, test_labels)*100,'%')

# use a file to make a prediction on the image
start_time = time.time()
use_file = input('enter filename: ')
prediction().text_prediction(use_file,clf,'SVM')
print (time.time() - start_time, "seconds")