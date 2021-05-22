from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def front(name=None):
    return render_template('front.html', name = name)

if __name__ == '__main__':
    app.run()