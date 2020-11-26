# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 23:24:11 2020

@author: Vrajesh
"""

import pyaudio
import wave
import os

 

def record():
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "audio.wav"
 
    audio = pyaudio.PyAudio()
    # start Recording
    #myrecording = sd.rec(RECORD_SECONDS * RATE, samplerate=RATE, channels=2)
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index=1,
                    frames_per_buffer=CHUNK)
    print ("recording speak...")
    frames = []
     
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording")
     
     
    # stop Recording

    stream.stop_stream()
    stream.close()
    audio.terminate()
     
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    
    path =os.path.abspath(WAVE_OUTPUT_FILENAME)
    return path
