import time
from extractor import letter_extractor

class letter_splitter:
    # user input for language: Bulgarian or english
    language = input('enter language: ')
    
    # Bulgarian true
    if language == 'Bulgarian':
        start_time = time.time()
        # set letter extractor to extract
        extract = letter_extractor()
        # array of training files
        training_files = [
            './languages/training/bulgarian/train1.png','./languages/training/bulgarian/train2.png','./languages/training/bulgarian/train3.png',
            './languages/training/bulgarian/train4.png','./languages/training/bulgarian/train5.png','./languages/training/bulgarian/train6.png',
            './languages/training/bulgarian/train7.png','./languages/training/bulgarian/train8.png'
            ]
        # Initializes name counter
        name_counter = 0

        # loop through training files one at a time
        for file in training_files:
            # directory of files to be stored
            directory1 = './languages/Bulgarian/training_type_test/small/'
            
            directory2 = './languages/Bulgarian/training_type_test/large/'
            # increment name counter each file by 1000
            name_counter = name_counter + 1000

            large_size = 200,200
            small_size = 20,50

            # call letter extract method from extractor
            letters = extract.extract_letters(directory1, file, name_counter,'Bulgarian', small_size)

            letters = extract.extract_letters(directory2, file, name_counter,'Bulgarian', large_size)
        # time of execution
        print (time.time() - start_time, "seconds" )

    # english true
    elif language == 'english':
        start_time = time.time()
        # set letter extractor to extract
        extract = letter_extractor()
        # array of training files
        training_files = [
            './languages/training/english/train1.png', './languages/training/english/train2.png', './languages/training/english/train3.png', './languages/training/english/train4.png',
            './languages/training/english/train5.png','./languages/training/english/train6.png','./languages/training/english/train7.png','./languages/training/english/train8.png',
            './languages/training/english/train9.png','./languages/training/english/train10.png','./languages/training/english/train11.png','./languages/training/english/train12.png',
            './languages/training/english/train13.png','./languages/training/english/train14.png','./languages/training/english/train15.png','./languages/training/english/train16.png',
            './languages/training/english/train17.png','./languages/training/english/train18.png'
            ]
        # Initializes name counter
        name_counter = 0
        
        # loop through training files one at a time
        for file in training_files:
            # directory of files to be stored
            directory1 = './languages/english/training_type_test/small/'
            directory2 = './languages/english/training_type_test/large/'
            # increment name counter each file by 100
            name_counter = name_counter + 1000
           
            large_size = 200,200
            small_size = 20,50
             # call letter extract method from extractor
            letters = extract.extract_letters(directory1, file, name_counter,'english', small_size)

            letters = extract.extract_letters(directory2, file, name_counter,'english', large_size)
        # time of execution
        print (time.time() - start_time, "seconds" )