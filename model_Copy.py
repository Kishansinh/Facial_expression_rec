from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog

import tensorflow as tf

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
print("file path is :::------------")
print(file_path)
img = image.load_img(file_path,target_size=(224,224), grayscale=True)
img=np.reshape(img,(224,224,1))

img = np.expand_dims(img, axis=0) 

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()

    def predict_emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        print(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)])
