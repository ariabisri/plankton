from flask import Flask, render_template, request
from readyolo import yolo_pred
from readkeras import keras_pred

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__, template_folder="templates")


@app.route('/')
@app.route('/index')
def index():
    
    return render_template('index.html')
    # return "Hello, World!"

@app.route('/result', methods=['POST'])
def result():
    try:
        model_type = request.form.get('model')
        img_path = request.files['file'].stream
        image_dis= image_display(img_path)
        op_image = Image.open(img_path)
        # img = preprocess_img(img_path)
        if request.method == 'POST':
            if model_type=='resnet':
                labels, scores = keras_pred(op_image)
            elif model_type == 'yolo':
                img_array = np.array(op_image)
                labels, scores = yolo_pred(img_array)
                # return render_template("result.html", prediksi=str(pred), img_path=img_dis, deb=deb, labels= labels, scores = scores, top = top, model_type = model_type )
            return render_template("result.html",  img_path=image_dis, labels= labels, scores = scores, model_type = model_type )
                
     
    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


def image_display(img):
    im = Image.open(img)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue()).decode('utf-8')
    return encoded_img_data

if __name__ == '__main__':
    app.run(debug=True)