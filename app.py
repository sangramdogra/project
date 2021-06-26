from flask import Flask, render_template, request, Response
import pickle
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
import imutils
from resources.preprocess import *
from werkzeug.utils import secure_filename
import os
from keras.models import load_model

model = pickle.load(open('Heat Exchanger/model.pkl', 'rb')) # Loading model pickle
ss = pickle.load(open('Heat Exchanger/s_scaler_train.pkl', 'rb')) # Loading standard scaler for train
invss = pickle.load(open('Heat Exchanger/s_scaler_train_y.pkl', 'rb')) # Loading standard scaler for y
model_li = pickle.load(open('li-ion/ridge.pkl','rb'))

model_ha = load_model('Human Activity/lstm_model.h5')
upload_dir = 'DIR'
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

@app.route('/human_activity')
def human_activity():
    return render_template('human_activity.html')

@app.route('/human_activity', methods = ["GET", "post"])
def activity_output():
    ACTIVITIES = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING',
}
    if request.method == "POST":
        file = request.files['input_file']
        file.save(os.path.join(upload_dir, secure_filename(file.filename)))
        data = pickle.load(open(upload_dir + '/' + file.filename, 'rb'))
        pred = np.argmax(model_ha.predict(data))
        return render_template('human_activity.html', output = ACTIVITIES[pred])



@app.route('/li')
def li():
    return render_template('liion.html')

@app.route('/li', methods = ['GET', 'POST'])
def function():
    if request.method == "POST":
        file = request.files['input_file']
        file.save(os.path.join(upload_dir, secure_filename(file.filename)))
        data = preprocess(upload_dir + '/' + file.filename)
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
    
@app.route('/self_driven')
def self_driven():
    return render_template('self_driven_car.html')

def automate():
    img = cv2.imread('steering_wheel_image.jpg')
    img = cv2.resize(img, (64, 64))
    i = -1 
    try:
        with open('autopilot/Autopilot-TensorFlow-master/driving_dataset/data.txt', 'r') as f:
            x = f.readlines()
            f.close()
        
        while (True):
            i+=1
            img2 = cv2.imread('autopilot/Autopilot-TensorFlow-master/driving_dataset/' + x[i].split(' ')[0])
            temp = img2[-64:,-64:]
            img1 = imutils.rotate(img, angle = -float(x[i].split(' ')[1].strip('\n')))
            weighted = cv2.addWeighted(img1, 1, temp, 0.1, 0.0)
            img2[-64:,-64:] = weighted
            _, frame_buff = cv2.imencode('.jpg', img2)
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame_buff) + b'\r\n')
    except:
        yield ('End Of Loop')
@app.route('/automation')
def automation():
    return Response(automate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    


if __name__ == '__main__':
    app.run(debug = True)
