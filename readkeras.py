from keras.api.applications.vgg16 import decode_predictions
from keras.api.models import load_model
from keras.api.utils import img_to_array
from PIL import Image
import numpy as np
from PIL import Image
import base64
import io

# vscode error
# from keras.models import load_model
# from keras.utils import img_to_array
# from keras.applications.vgg16 import decode_predictions
# from keras.applications.resnet50 import ResNet50

mdl = load_model('model/ResNet50.h5', compile=False)
# model2 = ResNet50()

def preprocess_img(op_img):
    # op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize)
    img_reshape = img2arr.reshape((1, img2arr.shape[0], img2arr.shape[1], img2arr.shape[2]))
    return img_reshape

def predict_result(img, model):
    pred = model.predict(img)
    pred =decode_predictions (pred)
    # pred = pred [0][0]
    return pred

def keras_pred(image):
    img = preprocess_img(image)
    pred = predict_result(img, mdl)
    num_result = len (pred[0])
    labels = []
    conf = []
    if num_result>3 :
        top = 3
    else :
        top =num_result

    for x in range(top):
        labels.append(pred[0][x][1])
        conf.append ('%.2f%%' % (pred[0][x][2]*100))
    
    
    return labels, conf