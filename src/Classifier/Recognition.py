"""
Author: Alfredo Sánchez Sánchez
Date: 25/06/2020
Project name: Music Detecting System (TFG)
Description: Final Degree Proyect Final project for the Telecomunication degree in ICAI Comillas, in Madrid (Spain) It consists on a system installed in raspbian using Unicorn Hat a 8x8 WS2812B RGB leds, the leds will shine at the same rithm and compass as the music that is sounding and will also detect the genre of the music that is being sounded. 
"""
# Import libraries
import librosa
import numpy as np
import tensorflow as tf
import argparse
    """
@description: Method to convert a list of songs to a np array of melspectrograms this will be the most important method at the time of obtaining the data we will need to fit the model
Inputs: Songs in the dataset, the number of fft that will be calculated, the number of samples between successive frames (hop_length)
Outputs: An array of the melspectrogram values
"""
def to_melspectrogram(songs, n_fft=1024, hop_length=256):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
        hop_length=hop_length, n_mels=128)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    # np.array([librosa.power_to_db(s, ref=np.max) for s in list(tsongs)])
    #print(np.array(list(tsongs)))
    return np.array(list(tsongs))
    """
@description: Method to split a song into multiple songs using overlapping windows this will help at the time of using differect length in time for the songs that will fit the model
Inputs: The window length (always 0.5), the overlap length (always 0.5), the X signal and the y genre
Outputs: Array of the new signal and the new array of genres (same as y but more9
"""
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
    """
@description: Method to call the other methods to split songs and obtaining the melspectrogram of the song recorded
Inputs: The song file .WAV
Outputs: The melspectrogram of the song
"""
def make_dataset_dl(song):
    # Convert to spectrograms and split into small windows
    signal, sr = librosa.load(song, sr=None)

    # Convert to dataset of spectograms/melspectograms
    signals = splitsongs(signal)

    # Convert to "spec" representation
    specs = to_melspectrogram(signals)

    return specs
    """
@description: Method to map the genres with a value of the map, for example, giving 1 mapping it to genre classical
Inputs: The number or key and the directory of genres, it will be declared in the main
Outputs: The genre
"""
def get_genres(key, dict_genres):
    # Transforming data to help on transformation
    labels = []
    tmp_genre = {v:k for k,v in dict_genres.items()}

    return tmp_genre[key]
    """
@description: Method to return the genre given the classifier output
Inputs: The scores of the classifier output and the directory of genres, it will be declared in the main
Outputs: The genre clasiffied and the probability of each other
"""
def majority_voting(scores, dict_genres):
    preds = np.argmax(scores, axis = 1)
    values, counts = np.unique(preds, return_counts=True)
    counts = np.round(counts/np.sum(counts), 2)
    votes = {k:v for k, v in zip(values, counts)}
    votes = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1], reverse=True)}
    return [(get_genres(x, dict_genres), prob) for x, prob in votes.items()]

if __name__ == '__main__':
    # We will decide to introduce the model and the audio file by command line or by default
    parser = argparse.ArgumentParser(description='Music Genre Recognition')
    parser.add_argument('-m', '--model', type=str, required=False)
    parser.add_argument('-s', '--song', type=str, required=False)
    args = parser.parse_args()
    if (args.model == None or args.song==None):
        song = '../../data/samples/Count.wav'
        model_file='../../data/Model.h5'
    else:
        song = args.song
        model_file=args.model

    # Constants
    genres = {
        'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
        'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9
    }

    X = make_dataset_dl(song)

    model = tf.keras.models.load_model(model_file)

    preds = model.predict(X)
    votes = majority_voting(preds, genres)
    print("{} is a {} song".format(song, votes[0][0]))
    print("most likely genres are: {}".format(votes[:3]))




