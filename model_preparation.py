# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 00:28:58 2020

@author: Vrajesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from librosa import display
import librosa
import noisereduce as nr
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras import models
from keras import optimizers
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.utils import plot_model


x_train=[]
x_test=[]
y_train=[]
y_test=[]

def preprocess(data):
    # perform noise reduction
    reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=data, verbose=False)
    #removing silence  
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=20,
                                      frame_length=512, hop_length=64)
    return trimmed



path='Speaker Recognition Dataset'
files = os.listdir('Speaker Recognition Dataset')
for name in tqdm(files):
    
  filename = path+"/"+name
  #print(filename)
  files1 = os.listdir(filename)
  for name1 in files1:
      #print(filename+"/"+name1)
      filename1 = filename+"/"+name1
      y,sr=librosa.load(filename1)
      y=preprocess(y)
      mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
      melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000).T,axis=0)
      chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
      chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=40).T,axis=0)
      chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=40).T,axis=0)
      features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(40,5))
      x_train.append(features)
      y_train.append(name)
      

     
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train1 = le.transform(y_train)

x_train=np.array(x_train)
y_train=np.array(y_train1)

x_train_2d=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
x_train_2d.shape
x_train, x_test, y_train, y_test = train_test_split(x_train_2d, y_train, test_size = 0.2, random_state = 0)

# x_train=x_train_2d
# x_test=x_test_2d
# y_train=y_train
# y_test=y_test

#converting to one hot
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
y_train.shape,y_test.shape

#reshaping to 2D 
x_train=np.reshape(x_train,(x_train.shape[0], 40,5))
x_test=np.reshape(x_test,(x_test.shape[0], 40,5))
x_train.shape,x_test.shape

#reshaping to shape required by CNN
x_train=np.reshape(x_train,(x_train.shape[0], 40,5,1))
x_test=np.reshape(x_test,(x_test.shape[0], 40,5,1))
x_train.shape,x_test.shape

#making model
model=Sequential()
model.add(Conv2D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(40,5,1)))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(10,activation="softmax",name='output'))

plot_model(model, to_file='CNNmodel.png')

#compiling
model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

#training the model
history = model.fit(x_train,y_train,batch_size=30,epochs=100,validation_data=(x_test,y_test))

import h5py
model.save('CNNmodel.h5')

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#train and test loss and scores respectively
train_loss_score=model.evaluate(x_train,y_train)
test_loss_score=model.evaluate(x_test,y_test)
print(train_loss_score)
print(test_loss_score)

