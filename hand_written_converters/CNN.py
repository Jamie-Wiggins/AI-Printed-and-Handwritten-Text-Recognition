# run on python 3.6 - all others on python 3.7.7
import os
import time
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from emnist import extract_training_samples
from emnist import extract_test_samples
from imageio import imread, imsave
import imageio as io

# extract letters from emnist data set
images, labels = extract_training_samples('letters')
data_slice = 60000
data_slice2 = 120000
# load training images
train_images = images[:data_slice,:,:]
# load training labels
train_labels = labels[:data_slice,]
# load testing images
test_images = images[data_slice:data_slice2,]
# load testing labels
test_labels = labels[data_slice:data_slice2,]
# reshape test images 
test_images = test_images.reshape(data_slice, 28, 28, 1)
# reshape train images
train_images = train_images.reshape(data_slice, 28, 28, 1)
test_labels = to_categorical(test_labels)
train_labels = to_categorical(train_labels)

# define cnn type
cnn = Sequential()
# layer 1
convolution1 = Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu')
cnn.add(convolution1)
# layer 2
convolution2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
cnn.add(convolution2)
# layer 3
pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool1)
# layer 4
convolution3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
cnn.add(convolution3)
# layer 5
convolution4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
cnn.add(convolution4)
# layer 6
pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_1)
# layer 7: 20% nuerons randomly deactivited
drop_layer1 = Dropout(0.2)
cnn.add(drop_layer1)
# layer 8:
flat_layer_0 = Flatten()
cnn.add(Flatten())
# layer 9:
dense = Dense(units=128, activation='relu', kernel_initializer='uniform')
cnn.add(dense)
# layer 10:
dense1 = Dense(units=64, activation='relu', kernel_initializer='uniform')
cnn.add(dense1)
# layer 11: output layer
output_layer = Dense(units=27, activation='softmax', kernel_initializer='uniform')
cnn.add(output_layer)
# Compile the classifier with stated parameters
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# fit the model with the training data
history = cnn.fit(train_images, train_labels,batch_size=32, epochs=5,validation_data=(test_images, test_labels))
# results of test data
scores = cnn.evaluate(test_images, test_labels, verbose=0)
# print accuracy to console of test data
print("Accuracy: %.2f%%" % (scores[1]*100))

# convert number value results into letters
letters = "abcdefghijklmnopqrstuvwxyz"
def label_result(number):
    number = int(number) - 1
    letter = letters[number]
    return letter

class prediction:
    # returns an array of accumlated zeros, representing rows/columns of black pixels
    def zeros(a, b):
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        if b == 1:
          ranges = np.where(absdiff == 1)[0].reshape(-1,2)
        else:
          ranges = np.where(absdiff == 1)[0]
        return ranges
    
    def text_prediction(self, file):
        text = ""
        # Read in image
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        # Convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_invert = 255-img_gray
        (thresh, binary_gray_invert) = cv2.threshold(gray_invert, 127, 255, cv2.THRESH_BINARY)
        # Invert image
        img_sharp = cv2.addWeighted(img_gray, 2.7, cv2.blur(img_gray, (5, 5)), 0, 1)
        img_gray_inverted = cv2.adaptiveThreshold(img_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # Finds average of pixel values in each row of the image
        row_means = cv2.reduce(img_gray_inverted, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
        # finds all the cutpoints for rows
        row_cutpoints = prediction.zeros(row_means, 0)
        # intialise words array
        words = []
        # loop through the row cutpoints
        for n,(start,end) in enumerate(zip(row_cutpoints, row_cutpoints[1:])):        
            # cut each on row cut points
            line = img[start:end]
            line_gray_inverted = img_gray_inverted[start:end]
            # only continues if there is word/letters in the row
            if cv2.countNonZero(line_gray_inverted):
                column_means = cv2.reduce(line_gray_inverted, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
                column_gaps = prediction.zeros(column_means, 1)
                column_gap_sizes = column_gaps[:,1] - column_gaps[:,0]
                column_cutpoints = (column_gaps[:,0] + column_gaps[:,1] - 1) // 2
                # 10 for nvidia file, 7 for other test files
                # adjust depending on text and image sie
                filtered_cutpoints = column_cutpoints[column_gap_sizes > 12]
                # loop through the column cutpoints for words
                for xstart,xend in zip(filtered_cutpoints, filtered_cutpoints[1:]):
                    # snip word on cutpoints and append to words array
                    snip = binary_gray_invert[start:end, xstart:xend]
                    words.append(snip)
        # loop through all the words in order of the word word array
        for word in words: 
            # find new column cutpoints for letters             
            column_means = cv2.reduce(word, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
            column_gaps = prediction.zeros(column_means, 1)
            column_gap_sizes = column_gaps[:,1] - column_gaps[:,0]
            column_cutpoints = (column_gaps[:,0] + column_gaps[:,1] - 1) // 2
            # column cutpoints for letters within the word
            filtered_cutpoints = column_cutpoints[column_gap_sizes >= 1]
             # array of cutpoints
            cut_here = []
            cut_here = column_gaps.flatten()
            # loop through the letter column cutpoints
            for xstart,xend in zip(cut_here, cut_here[1:]):
                # snip (cut) on the cut points
                snip = word[:,xstart:xend]
                # only continues if there is a letter in the image
                if cv2.countNonZero(snip):
                    contours = cv2.findContours(snip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    [x,y,w,h] = cv2.boundingRect(contours[0])
                    imgROI = snip[:, x:x+w]
                    imgROIResized = cv2.resize(imgROI,(28,28))
                    # predict the letter using the model
                    letters = "abcdefghijklmnopqrstuvwxyz"
                    result = cnn.predict_classes(imgROIResized.reshape(1, 28, 28, 1))
                    result = int(result) - 1
                    # convert the result to a lette
                    letter = letters[result]
                    # append the letter to the text
                    text += letter
            # space between each word
            text += " "
        print(text)

# file location for model to prediction on
start_time = time.time()
use_file = input('enter filename: ')
extract = prediction()
extract.text_prediction(use_file)
print (time.time() - start_time, "seconds")

## GRAPHS ## 

# box plot
from numpy import mean
from numpy import std
predict = cnn.evaluate(test_images, test_labels)
print(predict)
print(len(predict))
print(test_images.shape)
print(test_labels.shape)
print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(predict)*100, std(predict)*100, len(predict)))
# box and whisker plots of results
plt.boxplot(scores)

# confusion matrix 

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

pred = cnn.predict_classes(test_images, batch_size = 32, verbose = 0)
rounded_labels=np.argmax(test_labels, axis=1)
rounded_labels[1]

def plot_confusion_matrix(rounded_labels, pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(rounded_labels, pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(rounded_labels, pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(rounded_labels, pred, classes=rounded_labels,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(rounded_labels, pred, classes=rounded_labels, normalize=True,
                      title='Normalized confusion matrix')
disp = plt.gcf()
disp.set_size_inches(20, 15)
plt.savefig('SVM_dsfsdsfhand.png')
plt.show()