from flask import Flask, render_template, request, send_file, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import PIL
import pandas as pd
import os



app = Flask(__name__)

gender = {0: 'เพศหญิง', 1: 'เพศชาย'}

import sys
sys.path.append('../Web_app/templates/Age.h5')

from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

model1 = tf.keras.models.load_model('../Web_app/templates/Age.h5')
model1.make_predict_function()


sys.path.append('../Web_app/templates/Gender.h5')

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

model2 = tf.keras.models.load_model('../Web_app/templates/Gender.h5')
model2.make_predict_function()


def predict_Age1(Rt):
    # Read the image and preprocess it
    img = image.load_img(Rt, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape) 
    x /= 255.
    result = model1.predict(x)
    return result[0]

def predict_Gender1(Rt):
    # Read the image and preprocess it
    img = image.load_img(Rt, target_size=(150, 150))
    g = image.img_to_array(img)
    g = g.reshape((1,) + g.shape)
    g /= 255.
    result = model2.predict(g)
    print(result[0])
    print(result.argmax())
    return result[0], gender[result.argmax()]

def predict_Age2(Lt):
    # Read the image and preprocess it
    img = image.load_img(Lt, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape) 
    x /= 255.
    result = model1.predict(x)
    return result[0]

def predict_Gender2(Lt):
    img = image.load_img(Lt, target_size=(150, 150))
    g = image.img_to_array(img)
    g = g.reshape((1,) + g.shape)
    g /= 255.
    result = model2.predict(g)
    print(result[0])
    print(result.argmax())
    return result[0], gender[result.argmax()]
    
def create_cropped_images(img_path):
    # Read the image and create cropped versions
    img = Image.open(img_path)

    # Left crop
    frac = 0.6
    left = 0
    upper = 0
    right = img.size[0] - ((1 - frac)) * img.size[0] 
    bottom = img.size[1]
    cropped_left = img.crop((left, upper, right, bottom))
    Rt = f'static/Rt_{img_path.split("/")[-1]}'
    cropped_left.save(Rt)

    # Right crop
    left = img.size[0] * ((1 - frac))
    upper = 0
    right = img.size[0]
    bottom = img.size[1]
    cropped_right= img.crop((left, upper, right, bottom))
    flipped = cropped_right.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    Lt = f'static/Lt_{img_path.split("/")[-1]}'
    flipped.save(Lt)

    return Rt, Lt



# routes
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get a list of all uploaded files and process each one
        images = request.files.getlist('images')
        app.config['UPLOAD_FOLDER'] = '/root/WebApp/Web_app/static'
        results = []

        for img in images:
            # Save the image to a temporary file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            img.save(file_path)
            filename = os.path.basename(file_path)
            # img_path = 'static/' + filename
            img_path = 'static/' + filename


            # Predict the age and gender for the image
            Rt, Lt = create_cropped_images(img_path)

            age_pred = (predict_Age1(Rt)+predict_Age2(Lt))/2
            age_pred1 = np.around(age_pred[0] + 6, 2)
            print(age_pred1)
            years = int(age_pred1)
            months = round((age_pred1 - years) * 12)
            if months == 12:
                years += 1
                months = 0
            age_pred2 = f"{years} ปี {months} เดือน"
            print(age_pred2)

            result1, gender_pred1 = predict_Gender1(Rt)
            result2, gender_pred2 = predict_Gender2(Lt)

            if result1[0] > result1[1]:
                max_result1 = result1[0]
            else:
                max_result1 = result1[1]

            if result2[0] > result2[1]:
                max_result2 = result2[0]
            else:
                max_result2 = result2[1]

            if max_result1 > max_result2:
                gender_pred = gender_pred1
            else:
                gender_pred = gender_pred2

            # Add the result to a list of results
            results.append((filename, age_pred2 , gender_pred))

        # Write the predictions to a CSV file
        with open('static/predictions.csv', 'w') as f:
            f.write('ชื่อไฟล์, อายุ (ปี), เพศ\n')
            for result in results:
                f.write(','.join(str(x) for x in result) + '\n')


        df = pd.read_csv('static/predictions.csv')
        table = df.to_html()

        return render_template('test.html', table=table)

@app.route('/download')
def download_file():
    path = 'static/predictions.csv'
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)