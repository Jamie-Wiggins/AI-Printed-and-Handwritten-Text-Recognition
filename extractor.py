import os
import cv2
import imageio
import numpy as np

class letter_extractor:
    # returns an array of accumlated zeros, representing rows/columns of black pixels
    def zeros(a, b):
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        if b == 1:
          ranges = np.where(absdiff == 1)[0].reshape(-1,2)
        else:
          ranges = np.where(absdiff == 1)[0]
        return ranges

    # extracts letters from the image
    def extract_letters(self, directory, filename, name_counter, langauge, size):
        # English folder names
        letter_folder_names = [
            'a', 'Aa', 'b','Bb', 'c','Cc', 'd','Dd', 'e','Ee', 'f','Ff', 'g','Gg', 'h','Hh', 'i','Ii', 'j','Jj', 'k','Kk',
            'l','Ll', 'm','Mm', 'n','Nn', 'o','Oo', 'p','Pp', 'q','Qq', 'r','Rr', 's','Ss', 't','Tt', 'u','Uu', 'v','Vv', 
            'w','Ww', 'x','Xx', 'y','Yy', 'z', 'Zz'
        ]
        # bulgarian folder names
        bul_letter_folder_names = [
            'а','А1','б','Б1','в','В1','г','Г1','д','Д1','е','Е1','ж','Ж1','з','З1','и','И1','й','Й1','к','К1','л','Л1','м','М1','н','Н1','о','О1','п','П1','р','Р1','с','С1','т','Т1','у','У1',
            'ф','Ф1','х','Х1','ц','Ц1','ч','Ч1','ш','Ш1','щ','Щ1','ъ','Ъ1','ь','Ь1','ю','Ю1','я','Я1'
        ]

        # number folder names
        number_folder_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

        # if language bulgarian use bulgarian folders, if english use english folders
        if langauge == 'Bulgarian':
            folder_names =  bul_letter_folder_names
        elif langauge == 'english':
            folder_names =  letter_folder_names + number_folder_names
        
        # Initializes string counter
        string_counter = 0

        # Read in image
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        # Convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Invert image
        gray_invert = 255-img_gray
        (thresh, binary_gray_invert) = cv2.threshold(gray_invert, 127, 255, cv2.THRESH_BINARY)
        # Sharpen the image
        img_sharp = cv2.addWeighted(img_gray, 2.7, cv2.blur(img_gray, (5, 5)), 0, 1)
        img_gray_inverted = cv2.adaptiveThreshold(img_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # Finds average of pixel values in each row of the image
        row_means = cv2.reduce(img_gray_inverted, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
        # finds the cutpoints for each row
        row_cutpoints = letter_extractor.zeros(row_means, 0)
        
        # loops through each row of letters
        for n,(start,end) in enumerate(zip(row_cutpoints, row_cutpoints[1:])):
            # cut on row cutpoints
            line = img[start:end]
            line_gray_inverted = img_gray_inverted[start:end]
            # only continues if there is word/letters in the row
            if cv2.countNonZero(line_gray_inverted):
                # cut points for columns either side of letters
                column_means = cv2.reduce(line_gray_inverted, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).flatten()
                column_gaps = letter_extractor.zeros(column_means, 1)
                column_gap_sizes = column_gaps[:,1] - column_gaps[:,0]
                column_cutpoints = (column_gaps[:,0] + column_gaps[:,1] - 1) // 2
                # adjust depending on text and image sie
                filtered_cutpoints = column_cutpoints[column_gap_sizes > 2]
                # loop through the cutpoint for columns on eah row
                for xstart,xend in zip(filtered_cutpoints, filtered_cutpoints[1:]):
                    # string counter max 59 for bulgarain
                    if langauge == 'Bulgarian':
                        if string_counter > 59:
                            string_counter = 0
                    # string counter max 60 for english
                    elif langauge == 'english':
                        if string_counter > 60:
                            string_counter = 0
                    # snip the image on the contours found
                    snip = binary_gray_invert[start:end, xstart:xend]
                    # find countours of letters
                    contours = cv2.findContours(snip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    [x,y,w,h] = cv2.boundingRect(contours[0])
                    # only cut the width
                    imgROI = snip[:, x:x+w]
                    # resize the image - two size for each langauge
                    imgROIResized = cv2.resize(imgROI, (size))
                    
                    dirName = directory + folder_names[string_counter]
                    if not os.path.exists(dirName):
                        os.makedirs(directory + folder_names[string_counter])
                        print("New Directory " , dirName ,  " Created ")
                    else:    
                        print("Directory " , dirName ,  " already exists") 
                    # save the image
                    imageio.imsave(directory + folder_names[string_counter] + '/' + str(name_counter) + '_snippet.png', imgROIResized)
                    print('saved: /' + str(name_counter) + '_snippet.png')
                    string_counter += 1
                    name_counter += 1