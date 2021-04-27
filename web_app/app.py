from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

import cv2
from config import car_config as config
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
import numpy as np
import mxnet as mx
import argparse
import pickle
import imutils
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

args = {"prefix":"vggnet", "checkpoints":"checkpoints", "epoch":45}
le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read()) 
print("[INFO] loading pre-trained model...")
checkpointsPath = os.path.sep.join([args["checkpoints"],
        args["prefix"]])
model = mx.model.FeedForward.load(checkpointsPath,
        args["epoch"])      

# compile the model
model = mx.model.FeedForward(
        ctx=[mx.cpu(0)],
        symbol=model.symbol,
        arg_params=model.arg_params,
        aux_params=model.aux_params)

# initialize the image pre-processors
sp = AspectAwarePreprocessor(width=224, height=224)
mp = MeanPreprocessor(config.R_MEAN, config.G_MEAN, config.B_MEAN)
iap = ImageToArrayPreprocessor(dataFormat="channels_first")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload',methods=['post'])
def upload():
    if request.method == 'POST':
        img2 = request.files['file']
        (print(type(img2)))
        if img2:
                filename = secure_filename(img2.filename)
                img2.save(filename)
                image = cv2.imread(filename)
                orig = image.copy()
                orig = imutils.resize(orig, width=min(500, orig.shape[1]))
                image = iap.preprocess(mp.preprocess(sp.preprocess(image)))
                image = np.expand_dims(image, axis=0)

                # classify the image and grab the indexes of the top-5 predictions
                preds = model.predict(image)[0]
                idxs = np.argsort(preds)[::-1][:5]

                label = le.inverse_transform(idxs[0])
                label = label.replace(":", " ")
                label = "{}: {:.2f}%".format(label, preds[idxs[0]] * 100)
                cv2.putText(orig, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,255), 2)
                
                op_filename = 'output_' + filename

                cv2.imwrite('static/' + op_filename, orig)
                full_filename = os.path.join(app.config['UPLOAD_FOLDER'] , op_filename)
    return render_template("index.html",msg1 = "Prediction", msg2 = "Ready", user_img = full_filename)

if __name__ == "__main__":
    
    app.run(host='0.0.0.0', debug=False)
