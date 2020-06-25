from recorder import *
from time import clock
import time
import sys
import numpy
import keyboard
import unicornhat as unicorn
import os
import librosa
import numpy as np
import tensorflow as tf
import argparse
bpm_list = []
prev_beat = clock()
low_freq_avg_list = []
def predict():
    song = 'audio.wav'
    model_file='../TFG/custom_cnn_2d.h5'
    X = make_dataset_dl(song)
    model = tf.keras.models.load_model(model_file)
    preds = model.predict(X)
    votes = majority_voting(preds)
    return votes[0][0]
def to_melspectrogram(songs, n_fft=1024, hop_length=256):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
        hop_length=hop_length, n_mels=128)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    # np.array([librosa.power_to_db(s, ref=np.max) for s in list(tsongs)])
    #print(np.array(list(tsongs)))
    return np.array(list(tsongs))

def splitsongs(X, overlap=0.5):
    # Empty lists to hold our results
    temp_X = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = 33000
    offset = int(chunk * (1. - overlap))

    # Split the song and create new ones on windows
    spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        if s.shape[0] != chunk:
            continue

        temp_X.append(s)

    return np.array(temp_X)

def make_dataset_dl(song):
    # Convert to spectrograms and split into small windows
    signal, sr = librosa.load(song, sr=None)

    # Convert to dataset of spectograms/melspectograms
    signals = splitsongs(signal)

    # Convert to "spec" representation
    specs = to_melspectrogram(signals)

    return specs

def get_genres(key, dict_genres):
    # Transforming data to help on transformation
    labels = []
    tmp_genre = {v:k for k,v in dict_genres.items()}

    return tmp_genre[key]

def majority_voting(scores):
    preds = np.argmax(scores, axis = 1)
    values, counts = np.unique(preds, return_counts=True)
    counts = np.round(counts/np.sum(counts), 2)
    votes = {k:v for k, v in zip(values, counts)}
    votes = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1], reverse=True)}
    print(votes)
    return [(x, prob) for x, prob in votes.items()]

def seleccionar_colores(genero):
    if(genero == 0):
#Genero Metal
        ASCIIPIC = [
         "        "
        ," xxxxxx "
        ,"      x "
        ," xxxxxx "
        ," xxxxxx "
        ,"      x "
        ," xxxxxx "
        ,"        "]
        r = 0
        g = 0
        b = 255
    if(genero == 1):
#Genero Disco
        ASCIIPIC = [
         "        "
        ,"   xx   "
        ,"  x  x  "
        ," x    x "
        ," x    x "
        ," xxxxxx "
        ,"        "
        ,"        "]
        r = 241
        g = 33
        b = 219
    if(genero == 2):
#Genero clasico
        ASCIIPIC = [
         "        "
        ," xxxxxx "
        ," xx  xx "
        ," xx  xx "
        ,"     xx "
        ," xxxxxx "
        ," xx     "
        ," xx     "]
        r = 220
        g = 32
        b = 78
    if(genero == 3):
#Genero Hip Hop
        ASCIIPIC = [
         "        "
        ," xxxxxx "
        ," xxxxxx "
        ,"   xx   "
        ,"   xx   "
        ," xxxxxx "
        ," xxxxxx "
        ,"        "]
        r = 8
        g = 112
        b = 10
    if (genero == 4):
#Genero Jazz
        ASCIIPIC = [
         "        "
        ," xxxxx  "
        ," x    x "
        ," x    x "
        ," x    x "
        ," x   x  "
        ,"        "
        ,"        "]
        r = 22
        g = 55
        b = 128
    if (genero == 5):
#Genero Country
        ASCIIPIC = [
         "        "
        ,"  x  x  "
        ," x    x "
        ," x    x "
        ," x    x "
        ,"  x  x  "
        ,"   xx   "
        ,"        "]
        r = 240
        g = 128
        b = 9
    if (genero == 6):
#Genero Pop
        ASCIIPIC = [
         "        "
        ,"  xx    "
        ," x  x   "
        ," x  x   "
        ," x  x   "
        ," xxxxxx "
        ,"        "
        ,"        "]
        r = 242
        g = 162
        b = 141
    if (genero == 7):
#Genero Blues
        ASCIIPIC = [
         "        "
        ," xxxxxxx"
        ," x  x  x"
        ," x  x  x"
        ," x  x  x"
        ,"  xx xx "
        ,"        "
        ,"        "]
        r = 93
        g = 71
        b = 139
    if(genero == 8):
#Genero Reggae
        ASCIIPIC = [
         "   xx   "
        ,"  x  x  "
        ," x  xxx "
        ,"xxxxx  x"
        ,"xxxxx  x"
        ," x  xxx "
        ,"  x  x  "
        ,"   xx   "]
        r = 246
        g = 200
        b = 29
    if(genero == 9):
#Genero Rock
        ASCIIPIC = [
         "        "
        ," xxxxxx "
        ," x  x   "
        ," x  xx  "
        ," x  x x "
        ,"  xx    "        
        ,"        "
        ,"        "]
        r = 255
        g = 0
        b = 0
    return r, g , b, ASCIIPIC
def encender_luces(tiempo , genero):
    r, g , b, ASCIIPIC = seleccionar_colores(genero)
    for y in range(height):
            for x in range(width):
                    hPos = (y) % len(ASCIIPIC)
                    chr = ASCIIPIC[hPos][x]
                    if chr == ' ':
                        unicorn.set_pixel(x, y, 0, 0, 0)
                    else:
                        unicorn.set_pixel(x,y,int(r),int(g),int(b))
    unicorn.show()
    time.sleep(tiempo)
def apagar_luces(tiempo):
        unicorn.clear()
        unicorn.show()
        time.sleep(tiempo)    
def detect_beats(encendido, bpm_avg, genero):
    if not input_recorder.has_new_audio: 
        return

    # get x and y values from FFT
    xs, ys = input_recorder.fft()
    
    # calculate average for all frequency ranges
    y_avg = numpy.mean(ys)

    # calculate low frequency average
    low_freq = [ys[i] for i in range(len(xs)) if xs[i] < 1000]
    low_freq_avg = numpy.mean(low_freq)
    
    global low_freq_avg_list
    low_freq_avg_list.append(low_freq_avg)
    cumulative_avg = numpy.mean(low_freq_avg_list)
    
    bass = low_freq[:int(len(low_freq)/2)]
    bass_avg = numpy.mean(bass)
    #print("bass: {:.2f} vs cumulative: {:.2f}".format(bass_avg, cumulative_avg))
    
    # check if there is a beat
    # song is pretty uniform across all frequencies
    if (y_avg > 10 and (bass_avg > cumulative_avg * 1.5 or(low_freq_avg < y_avg * 1.2 and bass_avg > cumulative_avg))):
        global prev_beat
        curr_time = clock()
        print(encendido)
        if curr_time - prev_beat > 60/180: # 180 BPM max
            #print("beat")
            seconds= curr_time - prev_beat
            print(seconds)
            if(encendido==True):
                encender_luces(seconds, genero)
                encendido=False
            if(encendido==False):
                apagar_luces(seconds)
                encendido=True
            global bpm_list
            bpm = int(60 / (curr_time - prev_beat))
            #print(bpm)
            if len(bpm_list) < 4:
                if bpm > 60:
                    bpm_list.append(bpm)
            else:
                bpm_avg = int(numpy.mean(bpm_list))
                #print(bpm_avg)
                if abs(bpm_avg - bpm) < 35:
                    bpm_list.append(bpm)
            # reset the timer
            prev_beat = curr_time
        # shorten the cumulative list to account for changes in dynamics
    if len(low_freq_avg_list) > 50:
        low_freq_avg_list = low_freq_avg_list[25:]
        avg=int(numpy.mean(low_freq_avg_list))
        sec = float(1/bass_avg)
        print(1000*sec)
        if(1000*sec>3) :
            y_avg=9
        elif(encendido==True):
            encender_luces(1000*sec, genero)
            encendido=False
        if(encendido==False):
            apagar_luces(1000*sec)
            encendido=True
    # keep two 8-counsecs of BPMs so we can maybe catch tempo changes
    if len(bpm_list) > 24:
        bpm_list = bpm_list[8:]

    # reset song data if the song has stopped
    if y_avg < 10:
        print("new song")
        encendido=False
        apagar_luces(0.001)
        bpm_list = []
        low_freq_avg_list = []
        bpm_avg = 0
        input_recorder.close()
        os.system("python ../TFG/leeraudio1.py")
        genero = predict()
        input_recorder.start()

if __name__ == "__main__":
    os.system("python ../TFG/leeraudio1.py")
    genero = predict()
    unicorn.set_layout(unicorn.HAT)
    unicorn.rotation(0)
    unicorn.brightness(0.5)
    width,height=unicorn.get_shape()    
    print("Reticulating splines")
    time.sleep(.5)
    print("Enabled unicorn poop module!")
    time.sleep(.5)
    print("Pooping rainbows...")
    input_recorder = InputRecorder()
    input_recorder.start()
    encendido = True
    bpm_avg = 0
    while True:
        detect_beats(encendido, bpm_avg, genero)
    # clean up
        if (keyboard.is_pressed('p')):
            print("Se presiono, paramos")
            input_recorder.close()
            sys.exit()
            break
