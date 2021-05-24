from flask import Flask, render_template, request, Response
import pickle
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
model = pickle.load(open('Heat Exchanger/model.pkl', 'rb')) # Loading model pickle
ss = pickle.load(open('Heat Exchanger/s_scaler_train.pkl', 'rb')) # Loading standard scaler for train
invss = pickle.load(open('Heat Exchanger/s_scaler_train_y.pkl', 'rb')) # Loading standard scaler for y

def output(*args):
    arr = np.asarray([args])
    out = model.predict(arr)
    return invss.inverse_transform(out)








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
        string = request.form.get('textspace')
        arr = np.asarray(list(map(int, string.split(','))))
        arr = ss.transform(arr.reshape(1, -1))
        output = model.predict(arr)
        output = invss.inverse_transform(output)
        return render_template('heatexchanger.html', output= output)

@app.route('/auto')
def auto():
    return render_template('ac.html')

def cam():
    #Testing model to capture face recognition and output via html
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        faces, conf = cv.detect_face(frame)
        if faces != []:
            for face in faces:
                frame = cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (255, 0, 0))
        ret, frame_buff = cv2.imencode('.jpg', frame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame_buff) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(cam(), mimetype='multipart/x-mixed-replace; boundary=frame')
    

if __name__ == '__main__':
    app.run()
