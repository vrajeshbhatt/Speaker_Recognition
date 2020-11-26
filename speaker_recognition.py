# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:52:39 2020

@author: Vrajesh
"""

from keras import models
import noisereduce as nr
import librosa
import numpy as np
import speech_recognition as sr
import record_audio
import pyttsx3
import time

labels=['Chandresh','Dhruv','Jay','Jenish','Kenil','Madhavi','Mahek','Maunik','Naimish','Vrajesh']
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices') 
engine.setProperty('voice', voices[1].id)


def tts(path):
    r = sr.Recognizer()
    audio_file = sr.AudioFile(path)
    with audio_file as source:
        audio = r.record(source)
    text=r.recognize_google(audio)
    print(text)

def speak(audio): 
    engine.say(audio) 
    engine.runAndWait() 

def softmax(x):
    """ applies softmax to an input x"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    
new_model=models.load_model('CNNmodel.h5')

def predict(file):

    #file=record_audio.record() #it gives the path of recorded audio
    #text=tts(file) #convert speech to text
    y,sr=librosa.load(file)
    #y=preprocess(y)
    y = nr.reduce_noise(audio_clip=y, noise_clip=y,use_tensorflow=True, verbose=False)
    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000).T,axis=0)
    chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=40).T,axis=0)
    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(40,5))
    features=features.reshape(1,200)
    features=np.reshape(features,(features.shape[0], 40,5))
    features=np.reshape(features,(features.shape[0], 40,5,1))
    #result=model.predict(features)
    
    result=new_model.predict_classes(features)
    #result = new_model.predict_proba(features)
    #result= softmax(result)
    #result=np.argmax(result)
    #result = le.inverse_transform([result])
    if result[0] ==9 or result[0]==5:
        speak('Access Granted')
        print('Access Granted')
    else:
        speak('Access Denied')
        print('Access Denied')
        
    print(labels[result[0]])
    print(result)

if __name__ == '__main__':
    
    file = record_audio.record()
    start = time.time()
    predict(file)
    tts(file)
    stop=time.time()
    print(stop-start)
   
    


    