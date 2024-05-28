import base64
import io
i=5
import urllib.parse

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from Recognition import PreProcessing


from flask import Flask, render_template, request, redirect

app = Flask(__name__)

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

@app.route('/Preprocess',methods=['POST'])
def pre_process_screen():
    if request.method=="POST":
        imageFile = request.files.get('imageFile', '')

        #print(imageFile)
       # print(imageFile.read())
        #imageFile.save("./static/tempImage2.png")



        imageIO = io.BytesIO()

        imageFile.save(imageIO)

        # uri = 'data:image/png;base64,' + urllib.parse.quote(base64.b64encode(imageIO.getvalue()).decode('ascii'))
        #
        # return render_template('PreProcessing.html', uri=uri) #Test loading process

        imageIO.seek(0)
        file_bytes = np.asarray(bytearray(imageIO.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        image,resized = PreProcessing.intial_processing(img)

        image = PreProcessing.dialation_segmentation(image,resized)

        #cv2.imwrite("BESTIMAGE.png",img) # Test opencv

        return image



if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)