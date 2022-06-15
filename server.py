import os
from enum import Enum
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, redirect, url_for, render_template, request


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
            return f"{f.filename} is not a jpg/png file", 400
        return "hello"
        #inputImage = Image.open(BytesIO(f.read()))
        #outImage = rcnn.ProcessImage(inputImage)
        #imgIO = BytesIO()
        #outImage.save(imgIO, 'JPEG')
        #base64Str = base64.b64encode(imgIO.getvalue()).decode()

        #return base64Str, 200
    except Exception as e:
        return str(e), 400

#@app.errorhandler(404)
#def not_found(error):
#    return render_template('error.html'), 404

if __name__ == "__main__":
    app.run("0.0.0.0", port=8888)