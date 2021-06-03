from __future__ import print_function
from flask import Flask, render_template, request, jsonify, redirect
import cv2
from werkzeug.utils import secure_filename
import os
from flask.helpers import url_for
import numpy as np
import io
from PIL import Image
import base64
import cvlib as cv


app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/', methods = ['GET', 'POST'])
def process():
    if request.method == 'POST':
        file = request.files['img'].read()
        # print('Content Information of read file', file.file)
        # filename = secure_filename(file.filename)
        # file.save(os.path.join('../', filename))
        # cv2.VideoCapture(file)
        img = np.fromstring(file, np.uint8)
        img = cv2.imdecode(img,cv2.cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces, _ = cv.detect_face(img)
        if faces != []:
            for face in faces:
                img = cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (255, 0, 0), thickness= 2)
        #print('len of array is', len(img))
        img = Image.fromarray(img.astype("uint8"))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        return render_template('index.html', output=img_base64.decode('utf-8'))
        
        
        



if __name__ == '__main__':
    app.run(debug=True, use_reloader= True)



























if __name__ == '__main__':
    app.run()

