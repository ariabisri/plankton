from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image
from keras.applications.vgg16 import decode_predictions
from keras.applications.resnet50 import ResNet50
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
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            img_path = request.files['file'].stream
            model = load_model('model/ResNet50.h5', compile=False)
            # model2 = ResNet50()
            pred = predict_result(img, model)
            img_dis=image_display(img_path)

            deb = len(pred[0])
            labels = []
            scores = []
            if deb>3 :
                top = 3
            else :
                top =deb

            for x in range(top):
                labels.append(pred[0][x][1])
                scores.append ('%.2f%%' % (pred[0][x][2]*100))

            return render_template("result.html", prediksi=str(pred), img_path=img_dis, deb=deb, labels= labels, scores = scores, top = top )
 
            # model = load_model('model/ResNet50.h5')
            # img = preprocess_img(path)
            # result = predict_result (img, model)
            # return render_template('result.html', img_path='null', prediction= 'null')
     
    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


    

def preprocess_img(img_path):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize)
    img_reshape = img2arr.reshape((1, img2arr.shape[0], img2arr.shape[1], img2arr.shape[2]))
    return img_reshape

def predict_result(predict, model):
    pred = model.predict(predict)
    pred =decode_predictions (pred)
    # pred = pred [0][0]
    return pred

def image_display(img):
    im = Image.open(img)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue()).decode('utf-8')
    return encoded_img_data

if __name__ == '__main__':
    app.run(debug=True)