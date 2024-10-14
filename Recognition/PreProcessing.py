import cv2 as cv
import numpy as np

from Recognition import Model
from Recognition.Model import tf

def intial_processing(image):

    # First resize the image so that it's always 720x1280
    h, w, c = image.shape  # height width what's c? Color channels

    new_w = 720
    ar = w / h  # aspect ratio
    new_h = int(new_w / ar)

    image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)



    # Blur the image

    blurredImage = cv.bilateralFilter(image, 101, 30, 30)


    # Binarize the image

    grayImage = cv.cvtColor(blurredImage, cv.COLOR_BGR2GRAY)

    binarizedImage = cv.adaptiveThreshold(grayImage, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 17, 20)

    return binarizedImage,image

def dialation_segmentation(image,originalImage):

    originalImage2 = originalImage.copy()

    # Line Segmentation

    kernel = np.ones((20, 100), np.uint8)  # The number might need to be changed
    dialated = cv.dilate(image, kernel, iterations=1)


    (contours, hierarchy) = cv.findContours(dialated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    sorted_contours_lines = contours

    lineSegmented = image.copy()

    for ctr in sorted_contours_lines:

        if cv.contourArea(ctr) < 1000:  # removes noise but can be adjusted
            continue
        x, y, w, h = cv.boundingRect(ctr)
        cv.rectangle(originalImage, (x, y), (x + w, y + h), (0, 0, 250), 2)



    # Word Segmentation

    kernel = np.ones((25, 25), np.uint8)  # The number might need to be changed
    dialated2 = cv.dilate(image, kernel, iterations=1)


    # Store each words from top to bottom and right to left.

    words_list = []
    words_per_line = []

    for ctr in sorted_contours_lines[::-1]:

        if cv.contourArea(ctr) < 1000:  # removes noise but can be adjusted
            continue
        x, y, w, h = cv.boundingRect(ctr)
        line = dialated2[y:y + h, x:x + w]

        (cnt, hierarchy) = cv.findContours(line.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        sorted_contours_words = sorted(cnt, key=lambda cntr: cv.boundingRect(cntr)[0], reverse=True)

        lineNumber=0

        for word in sorted_contours_words:

            if cv.contourArea(word) < 600:  # removes noise
                continue
            x2, y2, w2, h2 = cv.boundingRect(word)
            words_list.append([x + x2, y + y2, x + x2 + w2, y + y2 + h2])
            lineNumber+=1

        words_per_line.append(lineNumber)



    # Store each word separately

    words = []

    for word in words_list:
        word = originalImage2[word[1]:word[3], word[0]:word[2]]
        words.append(word)


    #Binarize the Resized words

    gray_words = [cv.cvtColor(word,cv.COLOR_BGR2GRAY) for word in words]

    binary_words = [cv.adaptiveThreshold(word,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,17, 10) for word in gray_words]

    predicted_text =""

    wordNumber=0
    lineNumber=0

    for word in binary_words:
        preds=  Model.predict2(word)
        prediction = ''.join(Model.decode_batch_predictions(preds))
        predicted_text+= prediction+" "
        wordNumber+=1

        if wordNumber==words_per_line[lineNumber]:
            predicted_text += "\n"

            wordNumber = 0
            lineNumber += 0


    return predicted_text