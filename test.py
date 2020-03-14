# -*- coding:utf-8 -*-
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import librosa
from keras.models import load_model
import numpy as np

model = load_model('best_model.hdf5')
classes = ["down","go","left","no","off","on","right","stop","up","yes"]

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

files = "0.wav"
samples,sample_rate = librosa.load(files,sr=16000)
samples = librosa.resample(samples,sample_rate,8000)
print(predict(samples))
