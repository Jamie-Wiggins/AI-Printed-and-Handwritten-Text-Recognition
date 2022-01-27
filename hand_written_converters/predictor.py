import os
import time
from imageio import imread, imsave
import numpy as np
import cv2
import imageio as io

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
    
    def text_prediction(self,file,model,modal_type):
        text = ""
        # Read in image
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        # Convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Invert image
        gray_invert = 255-img_gray
        (thresh, binary_gray_invert) = cv2.threshold(gray_invert, 127, 255, cv2.THRESH_BINARY)
        # Sharpen the image
        img_sharp = cv2.addWeighted(img_gray, 2.7, cv2.blur(img_gray, (5, 5)), 0, 1)
        # Apply adaptive thresholding
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
                filtered_cutpoints = column_cutpoints[column_gap_sizes > 7]
                # loop through the column cutpoints for words
                for xstart,xend in zip(filtered_cutpoints, filtered_cutpoints[1:]):
                    # snip word on cutpoints and append to words array 
                    snip = binary_gray_invert[start:end, xstart:xend]
                    words.append(snip)
        # loop through all the words in order of the word word array
        for word in words:  
            # find new column cutpoints for letters          
            column_means = cv2.reduce(word, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
            column_gaps = prediction.zeros(column_means,1)
            column_gap_sizes = column_gaps[:,1] - column_gaps[:,0]
            column_cutpoints = (column_gaps[:,0] + column_gaps[:,1] - 1) // 2
            # column cutpoints for letters within the word
            filtered_cutpoints = column_cutpoints[column_gap_sizes >= 1]
            # array of cutpoints
            cut_here = []
            cut_here = column_gaps.flatten()
            # loop through the letter column cutpoints
            for xstart,xend in zip(cut_here, cut_here[1:]):
                snip = word[:,xstart:xend]
                if cv2.countNonZero(snip):
                    contours = cv2.findContours(snip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    [x,y,w,h] = cv2.boundingRect(contours[0])
                    imgROI = snip[:, x:x+w]
                    if model_type == 'SVM':
                        imgROIResized = cv2.resize(imgROI, (28,28))
                        # PREDICT THE SNIPPET USING THE MODEL
                        hog_features = hog(imgROIResized, orientations=12, pixels_per_cell=(2, 2), cells_per_block=(1, 1))
                        result = model.predict(hog_features.reshape(1,-1))    
                        letters = "abcdefghijklmnopqrstuvwxyz"
                        result = int(result) - 1
                        letter = letters[result]
                        text += letter     
                    elif model_type == 'KNN':
                        imgROIResized = cv2.resize(imgROI,(28,28))
                        # PREDICT THE SNIPPET USING THE MODEL
                        letters = "abcdefghijklmnopqrstuvwxyz"
                        result = model.predict(imgROIResized.reshape(-1,784))
                        result = int(result) - 1
                        letter = letters[result]
                        text += letter
            text += " "
        print(text)