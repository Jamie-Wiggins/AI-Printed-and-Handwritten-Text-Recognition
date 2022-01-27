import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageFilter
import skimage
from skimage.viewer import ImageViewer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from emnist import extract_training_samples
from predictor import prediction

# load in sample from emnist
train_images, train_labels = extract_training_samples('letters')
test_images, test_labels = extract_training_samples('letters')
# create data slices 70/30 split
data_slice = 70000
data_slice2 = 100000
# load in training data
train_images = train_images[:data_slice,:,:]
train_labels = train_labels[:data_slice,]
# load in test data
test_images = test_images[data_slice:data_slice2,]
test_labels = test_labels[data_slice:data_slice2,]

# bit shift kernel - not used - has a negative effect on score
kernel = np.zeros( (9,9), np.float32)
kernel[3,7] = 1.0

new_test_images = []
for i in range(len(test_images)):
    #custom = cv2.filter2D(test_images[i], -1, kernel, borderType = cv2.BORDER_CONSTANT)
    # remove noise
    dst = cv2.blur(test_images[i], (5,5))
    #blur
    blurred = skimage.filters.gaussian(dst, sigma=(0.9, 0.9), truncate=3.5, multichannel=True)
    new_test_images.append(blurred)
    
new_test_images = np.array(new_test_images)

new_train_images = []
for i in range(len(train_images)):
    #custom = cv2.filter2D(train_images[i], -1, kernel, borderType = cv2.BORDER_CONSTANT)
    dst = cv2.blur(train_images[i], (5,5))
    blurred = skimage.filters.gaussian(dst, sigma=(0.9, 0.9), truncate=3.5, multichannel=True)
    new_train_images.append(blurred)

new_train_images = np.array(new_train_images)

# load new data and reshape
train_images = new_train_images[:,:].reshape(-1, 784).astype(np.float32)
test_images = new_test_images[:,:].reshape(-1, 784).astype(np.float32)
train_labels = np.array(train_labels, np.float32)
test_images = np.array(test_images, np.float32)
test_labels = np.array(test_labels, np.float32)

# create classifier
knn_clf = KNeighborsClassifier(n_neighbors=6)
# train the model
knn_clf.fit(train_images, train_labels)
# print out the score of testing to console
acc = knn_clf.score(test_images, test_labels)*100
print(acc,'%')

# convert number value results into letters
start_time = time.time()
use_file = input('enter filename: ')
prediction().text_prediction(use_file,knn_clf,'KNN')
print (time.time() - start_time, "seconds")

#confusion matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(knn_clf, test_images, test_labels,
                                 display_labels="abcdefghijklmnopqrstuvwxyz",
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
    
    disp = plt.gcf()
    disp.set_size_inches(20, 15)
plt.savefig('SVM_hand.png')