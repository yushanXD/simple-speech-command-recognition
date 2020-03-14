# -*- coding:utf-8 -*
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import librosa
import scipy
import os
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings
import random
from keras.models import load_model
import json
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

train_audio_path = "tensorflow-speech-recognition-challenge/"
labels = ["yes","no","up","down","left","right","on","off","stop","go"]
all_wave = []
all_label = []
for label in labels:
    print(label)
    waves =  [f for f in os.listdir(train_audio_path+'/'+label) if f.endswith('.wav')]
    for wav in waves:
        samples,sample_rate = librosa.load(train_audio_path+'/'+label+'/'+wav,sr=16000)
        samples = librosa.resample(samples,sample_rate,8000)
        if(len(samples)==8000):
            all_wave.append(samples)
            all_label.append(label)

le = LabelEncoder()
y = le.fit_transform(all_label)
classes = list(le.classes_)
print(classes)
y = np_utils.to_categorical(y,num_classes=len(labels))
all_wave = np.array(all_wave).reshape(-1,8000,1)
x_tr,x_val,y_tr,y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size=0.2,random_state=777,shuffle=True)
print(len(x_tr))
print(len(x_val))
print(len(y_tr))
print(len(y_val))

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

filepath = 'best_model.hdf5'
model.save(filepath) #saves the weights of the model as a HDF5 file.

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

model=load_model('best_model.hdf5')

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]


index=random.randint(0,len(x_val)-1)
samples=x_val[index].ravel()
print("Audio:",classes[np.argmax(y_val[index])])
ipd.Audio(samples, rate=8000)
print("Text:",predict(samples))
