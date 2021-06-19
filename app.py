import cv2
import sys
import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(150,150))
    img = np.reshape(img,[-1,150,150,1])
       

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict_classes(img)
    preds = preds.reshape(1,-1)[0]
    print(preds)

    if preds[0]==0:
        preds="Pneumonia Detected"
    elif preds[0]==1:
        preds="Normal"
       
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)