# from keras.preprocessing.image import img_to_array
# from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import Flask, request, redirect, url_for, render_template, jsonify, json
import io
from werkzeug.utils import secure_filename
import os
import cv2
# import sys
# import glob
import time
# import sys
from api import demo
from mtcnn import MTCNN
from os.path import join, dirname, realpath

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static')
# UPLOAD_FOLDER = os.path.join(home_dir, 'static')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    # print('Bắt đầu trang điểm nè...')
    if request.method == 'POST':
        content = request.get_data(as_text = True)
        content = str(content)
        prediction, duration = demo.test(os.path.join(app.config['UPLOAD_FOLDER'], content))
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], content), cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))
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
            img = cv2.cvtColor(cv2.imread(image_location), cv2.COLOR_BGR2RGB)
            img_w, img_h = img.shape[0], img.shape[1]
            detector = MTCNN()
            faces=detector.detect_faces(img)
            if len(faces) == 0:
                feedback = 'Không tìm thấy khuôn mặt nào... Trang điểm MÙ MỜ'
                imageFace= cv2.resize(img, (350, 350))
            else:
                if len(faces)==1:
                    feedback = '<b>1</b> khuôn mặt, bắt đầu thực hiện trang điểm...'
                else:
                    feedback = 'Úm ba la xì bùa, chọn <b>1</b> khuôn mặt và trang điểm...'
                faces = faces[0]['box']
                x, y, w, h = faces[0], faces[1], faces[2], faces[3]
                ext = [w, h][np.argmax([w, h])]
                ext=int(ext*1.15)
                x=int(x-0.5*int(ext-w))
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                ext=int(np.min([y+ext, x+ext, img_w, img_h]))
                imageFace= img[y:y+ext, x:x+ext, :]
                imageFace = cv2.resize(imageFace, dsize=(350, 350))
            im = Image.fromarray(imageFace)
            face_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            im.save(face_location)
            face_location = os.path.join('./static', filename)

            return jsonify(original_image = filename, face_image = face_location, msg = feedback, success = True)
                
        else:
            feedback = 'Chỉ chấp nhận file ảnh có định dạng <b>JPG, PNG</b> nha. Vui lòng thử lại ạ.'
            return jsonify(msg = feedback, success = False)
                                        
if __name__ == "__main__":

    print('Starting server :D')
    app.run(host = "0.0.0.0", port = 1024, debug = False, threaded = False)
