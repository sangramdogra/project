from flask import Flask, render_template, request
import pickle
import numpy as np

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

if __name__ == '__main__':
    app.run()
