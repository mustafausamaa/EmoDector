# Import necessary libraries
import os
import librosa
import librosa.core
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

from sklearn.model_selection import KFold

import pickle
from sklearn.preprocessing import MinMaxScaler

from python_speech_features import mfcc
from scipy.io import wavfile



# Function to extract feature extraction from the audio sample
def ex_features(X, sample_rate: float) -> np.ndarray:

    # Extract Short-Time Fourier Transform
    # Represents the frequency content of the input signal over time
    # Take the absolute value of the complex STFT coefficients
    stft = np.abs(librosa.stft(X))

    # Estimate the pitch of the stft extracted
    # Fmin and Fmax are the pitch frequencies to search for
    # returns 2 lists pitches and magnitudes.
    # pitches contains the pitch frequencies estimated for each time frame.
    # magnitudes contains the corresponding magnitudes (or strengths) of those pitch estimates.
    pitches, magnitudes = librosa.piptrack(
        y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)

    # Extract the pitch estimates with the highest magnitudes for each time frame
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    # Estimate the tuning offset of the pitch
    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    # Mean
    pitchmean = np.mean(pitch)
    # Standard deviation
    pitchstd = np.std(pitch)
    # Max
    pitchmax = np.max(pitch)
    # Min
    pitchmin = np.min(pitch)

    # Spectral centroid of the audio signal
    # Spectral centroid is a measure of the "center of mass" of the power spectrum
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    # Normalize to ensure that the values sum to 1
    cent = cent / np.sum(cent)
    # Mean
    meancent = np.mean(cent)
    # Std
    stdcent = np.std(cent)
    # Max
    maxcent = np.max(cent)

    # Calculate the spectral flatness
    # Spectral flatness is a measure of how "flat" or "peaked" is the power spectrum of the signal
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # Mel-frequency cepstral coefficients (MFCCs) of the audio signal
    # Calculate mean, std and max
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    # Calculate chroma features of the audio signal from the Short-time Fourier
    # Represents the distribution of pitch classes (the twelve notes in an octave)
    # chroma list contains the mean value of each of the twelve pitch classes
    chroma = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)

    # Calculate the Mel-scaled spectrogram feature of the audio signal
    mel = np.mean(librosa.feature.melspectrogram(
        y=X, sr=sample_rate).T, axis=0)

    # calculates the spectral contrast feature of the audio signal from the Short-time Fourier Transform (STFT) coefficients
    contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)

    # This code block computes the zero-crossing rate feature of the audio signal X using the librosa.feature.zero_crossing_rate function. The zero-c
    # rossing rate is a measure of the number of times that the audio signal crosses the zero-axis per second, and is often used as an indicator of
    # the amount of high-frequency content or noisiness in the signal.

    # Calculates the zero-crossing rate of the audio signal
    # A measure of the number of times that the audio signal crosses the zero-axis per second.
    # used to obtain the amount of high-frequency content or noisiness in the signal.
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    # Calculate magnitude features
    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # RMS of the magnitudes
    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    # Concatenate all the single valued features
    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    # Concatenate the single values to the other feature lists
    ext_features = np.concatenate(
        (ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features


#Function to normalizate the audio samples
def normalize_sample(sample, min_list, max_list):

    for i in range(len(sample)):
        sample[i] = sample[i]-min_list[i]/max_list[i]-min_list[i]

    return sample



#Function to read the normalization info of the used dataset
def read_normalization_file():
    with open("Audio/normalization.txt", "r") as file:
        lines = file.readlines()


        min_list = []
        max_list = []

        # Remove leading/trailing whitespaces and cast lines to float32
        # float_list = [float(line) for line in lines]

        for line in lines:
            # Split the line into separate values based on spaces
            values = line.strip().split()
        # print(values[0],values[1])
            min_list.append(float(values[0]))
            max_list.append(float(values[1]))
        
        return min_list,max_list


#Function to load the pickled machine learning model
def load_model(path):
    with open(path, 'rb') as f:
        model= pickle.load(f)
    return model



#Main function
def main():
    
   #Audio file path
   audio_sample_path = "Audio/audio_files/Recording(2).wav"
   
   #Load the audio file
   audio, sr = librosa.load(audio_sample_path, sr=22050)
   
   #Read normalization file
   min_list,max_list=read_normalization_file()
   
   #Feature extraction
   features=ex_features(audio,sr)
    
   #Feature normalization
   features_norm = normalize_sample(features, min_list, max_list)
   features_norm = features_norm.reshape(1, -1)


   #Model path
   model_path='Audio/sound_models/svm_model81.6_s+n+c.pkl'
    
   #Load the machine learning mode
   model=load_model(model_path)

   #Predict
   prediction = model.predict(features_norm)
   
   
   print(prediction)




if __name__ == "__main__":
    main()
