from flask import Flask, render_template, request, Response
import pickle
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from resources.preprocess import *
from werkzeug.utils import secure_filename

model = pickle.load(open('Heat Exchanger/model.pkl', 'rb')) # Loading model pickle
ss = pickle.load(open('Heat Exchanger/s_scaler_train.pkl', 'rb')) # Loading standard scaler for train
invss = pickle.load(open('Heat Exchanger/s_scaler_train_y.pkl', 'rb')) # Loading standard scaler for y
model_li = pickle.load(open('li-ion/ridge.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def front(name=None):
    return render_template('front.html', name = name)

@app.route('/he')
def he():
    return render_template('heatexchanger.html')

@app.route('/he', methods = ['POST'])
def fun():
    if request.method == "POST":
        string = request.form.to_dict()
        string = list(string.values())[:-1]
        string = [float(x) if x!=''  else 0 for x in string]
        arr = np.asarray(string).reshape(1, -1)
        arr = ss.transform(arr)
        output = model.predict(arr)
        output = invss.inverse_transform(output)
        return render_template('heatexchanger.html', output= output)

@app.route('/li')
def li():
    return render_template('liion.html')

@app.route('/li', methods = ['GET', 'POST'])
def function():
    if request.method == "POST":
        file = request.files['input_file']
        file.save(secure_filename(file.filename))
        data = preprocess(file.filename)
        pred= model_li.predict(data)
        return render_template('liion.html', output = pred[0])


@app.route('/auto')
def auto():
    return render_template('ac.html')

def cam():
    #Testing model to capture face recognition and output via html
    capture = cv2.VideoCapture(0)
    while True:
        _, frame = capture.read()
        faces, _ = cv.detect_face(frame)
        if faces != []:
            for face in faces:
                temp = frame[face[1]:face[3], face[0]:face[2]]
                label, confidence = cv.detect_gender(temp)
                frame = cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (255, 0, 0))
                frame = cv2.putText(frame, label[np.argmax(confidence)], (face[2], face[3]+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        _, frame_buff = cv2.imencode('.jpg', frame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame_buff) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(cam(), mimetype='multipart/x-mixed-replace; boundary=frame')
    

if __name__ == '__main__':
    app.run(debug = True)
