import os
from io import BytesIO
from PIL.Image import open 
from flask import Flask, redirect, url_for, render_template, request
from recognize import Recognize

app = Flask(__name__)


@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/v1/index')
def home():
    return render_template('index.html')

def checkFileType(filename, allow=['.jpeg', '.jpg', '.png']):
    ext = os.path.splitext(filename)[1].lower()
    return ext in allow

@app.route('/upload', methods=['POST'])
def upload():
    try:
        f = request.files['file']
        if not checkFileType(f.filename):
            return 'error'#f"{f.filename} is not a jpg/png file", 400
        im = open(BytesIO(f.read()))
        reg = Recognize()
        return reg.recognize(im)

    except Exception as e:
        return str(e), 400
if __name__ == "__main__":
    app.run("0.0.0.0", port=8888)