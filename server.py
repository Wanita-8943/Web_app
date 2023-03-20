from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import PIL

app = Flask(__name__)

gender = {0: 'Female', 1: 'Male'}

import sys
sys.path.append('/root/WebApp/Web_app/templates/Age.h5')

from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

model1 = tf.keras.models.load_model('/root/WebApp/Web_app/templates/Age.h5')
model1.make_predict_function()


sys.path.append('/root/WebApp/Web_app/templates/Gender.h5')

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})
model2 = tf.keras.models.load_model('/root/WebApp/Web_app/templates/Gender.h5')


model2.make_predict_function()

def predict_image1(img_path):
    # Read the image and preprocess it
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape) 
    x /= 255.
    result = model1.predict(x)
    return np.around(result[0]+6, 2)

def predict_image2(img_path):
    # Read the image and preprocess it
    img = image.load_img(img_path, target_size=(150, 150))
    g = image.img_to_array(img)
    g = g.reshape((1,) + g.shape) 
    g /= 255.
    result = model2.predict(g)
    return gender[result.argmax()]

# routes
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Read the uploaded image and save it to a temporary file
        file = request.files['image']
        img_path = 'static/panoramic.jpg'
        file.save(img_path)

        frac = 0.60
        filename = img_path.split("/")[-1]

        # Read the image and create cropped versions
        img = Image.open(img_path)

        #left---------------------------------------------------
        left = 0
        upper = 0
        right = img.size[0]-((1-frac))*img.size[0] 
        bottom = img.size[1]
        cropped_left = img.crop((left, upper, right, bottom))
        #save setting by youreself
        cropped_left.save('static/''Rt_'+filename)

        #right ---------------------------------------------------
        left = img.size[0]*((1-frac))
        upper = 0
        right = img.size[0]
        bottom = img.size[1]
        cropped_right= img.crop((left, upper, right, bottom))
        #save setting by youreself
        cropped_right.save('static/unflip_'+filename)

        #flip ---------------------------------------------------
        #read the image
        im = Image.open('static/unflip_'+filename)

        #flip image
        out = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        out.save('static/Flip_'+filename)
        

        # Predict the age
        age_pred = predict_image1(img_path)
        gender_pred = predict_image2(img_path)

        # Render the prediction result
        return render_template('test.html', prediction1=age_pred[0], prediction2=gender_pred)



if __name__ == '__main__':
    app.run(debug=True)