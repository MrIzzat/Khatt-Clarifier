import base64
import io
import os
import time

i=5
import urllib.parse

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from Recognition import PreProcessing

from threading import Thread


from flask import Flask, render_template, request, redirect, send_from_directory, url_for

app = Flask(__name__)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
@app.route('/', methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        if request.method == "POST":
            return "Error, POST request to index page"
        else:
         return "Error loading main screen"

@app.route('/CropperScreen',methods=['POST'])
def cropper_screen():
    if request.method == "POST":
        imageFile = request.files.get('imageFile', '')

        #imageFile.save("./static/tempImage.png")

        imageIO = io.BytesIO()

        imageFile.save(imageIO)

        uri = 'data:image/png;base64,' + urllib.parse.quote(base64.b64encode(imageIO.getvalue()).decode('ascii'))

        return render_template('CropperScreen.html',uri=uri)
    else:
        return "Error while accessing cropper page"

def doRecognition(imageFile):

    imageIO = io.BytesIO()

    imageFile.save(imageIO)

    # uri = 'data:image/png;base64,' + urllib.parse.quote(base64.b64encode(imageIO.getvalue()).decode('ascii'))
    #
    # return render_template('PreProcessing.html', uri=uri) #Test loading process

    imageIO.seek(0)
    file_bytes = np.asarray(bytearray(imageIO.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image, resized = PreProcessing.intial_processing(img)

    image = PreProcessing.dialation_segmentation(image, resized)

    # cv2.imwrite("BESTIMAGE.png",img) # Test opencv

    return image

@app.route('/RecognizeText',methods=['POST'])
def recognize_text():
    if request.method=="POST":

        imageFile = request.files.get('imageFile', '')

        text = doRecognition(imageFile)


        return render_template('RecognizedText.html',text=text)



if __name__=="__main__":

    app.run(host="0.0.0.0", port=5000,debug=True)
