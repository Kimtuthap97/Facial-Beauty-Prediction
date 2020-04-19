# from keras.preprocessing.image import img_to_array
# from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import Flask, request, redirect, url_for, render_template, jsonify, json
import io
# from keras.models import load_model
from werkzeug.utils import secure_filename
import os

import cv2
import sys
import glob
import time
import sys
from api import demo

# initialize our Flask application and the Keras model
from os.path import join, dirname, realpath

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static')
# UPLOAD_FOLDER = os.path.join(home_dir, 'static')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#cascPath = "haarcascade_frontalface_default.xml"
cascPath = os.path.join(app.config['UPLOAD_FOLDER'], 'haarcascade_frontalface_default.xml')
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    full_filename = os.path.join('./static', 'placeholder.png')
    print(full_filename)
    return render_template('index.html', displayedimage = full_filename)
    
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    print('Bắt đầu trang điểm nè...')
    if request.method == 'POST':
        content = request.get_data(as_text = True)
        content = str(content)
        prediction, duration = demo.test(os.path.join(app.config['UPLOAD_FOLDER'], content))
        prediction = cv2.normalize(prediction, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        prediction = prediction.astype(np.uint8)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], content), cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))
        # print('Done in {} s'.format(duration))
        feedback = 'Chương trình mất <b>{}s</b> để hoàn thành trang điểm'.format(duration)
        return jsonify(msg = feedback)
    
@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            feedback = 'Vui lòng chọn một file ảnh.'
            return jsonify(msg = feedback, success = False)
        file = request.files['file']
        if file.filename == '':
            feedback = 'Ảnh không có tên. Vui lòng thử lại với ảnh có tên ạ.'
            return jsonify(msg = feedback, success = False)
        if file and allowed_file(file.filename):
            millis = int(round(time.time() * 1000))
            millis = str(millis)
            filename = millis + secure_filename(file.filename)
            f = request.files['file']
            image_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(image_location)
            # print('Saved upload file :D')
            face_filename = filename
                       
            imageFace = cv2.imread(image_location)
            if imageFace is not None:
                gray = cv2.cvtColor(imageFace, cv2.COLOR_BGR2GRAY)
            
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            print("Found {0} faces!".format(len(faces)))
            if len(faces) == 1:
                feedback = '1 khuôn mặt, bắt đầu thực hiện trang điểm...'
                for (x, y, w, h) in faces:
                    ext = [w, h][np.argmax([w, h])]
                    ext=int(ext*1.2)
                    x=int(x-0.5*int(ext-w))
                    if x < 0:
                        x =0
                    if y < 0:
                        y=0
                    imageFace= imageFace[y:y+ext, x:x+ext, :]

            elif len(faces) == 0:
                feedback = 'Không tìm thấy khuôn mặt nào... Trang điểm MÙ MỜ'
            else:
                feedback = 'Úm ba la xì bùa, chọn 1 khuôn mặt và trang điểm...'
                for (x, y, w, h) in faces[0:1]:
                    ext = [w, h][np.argmax([w, h])]
                    ext=int(ext*1.2)
                    x=int(x-0.5*int(ext-w))
                    if x < 0:
                        x =0
                    if y < 0:
                        y=0
                    imageFace= imageFace[y:y+ext, x:x+ext, :]
            imageFace = cv2.resize(imageFace, dsize=(350, 350))
            imageFace = cv2.cvtColor(imageFace, cv2.COLOR_BGR2RGB)
            
            im = Image.fromarray(imageFace)
            face_location = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
            im.save(face_location)
            face_location = os.path.join('./static', face_filename)

            return jsonify(original_image = filename, face_image = face_location, msg = feedback, success = True)
                
        else:
            feedback = 'Chỉ chấp nhận file ảnh có định dạng <b>JPG, PNG</b> nha. Vui lòng thử lại ạ.'
            return jsonify(msg = feedback, success = False)
                                        
if __name__ == "__main__":

    print('Starting server :D')
                
    # model = load_model('my_model.h5')
    
    # print('model loaded')
        
    # with open('All_labels_alphabetized_nolabel.txt', 'r') as f:
    #         x = f.read().splitlines()
    #         x = [float(i) for i in x]
    #         x = [x * 2 for x in x]

    app.run(host = "0.0.0.0", port = 1024, debug = False, threaded = False)
