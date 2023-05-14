import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import os
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
import pickle
from statistics import mode
import librosa
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class GenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GenreClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def extract_features(data, sampling_rate):
    #Read audio file
    hop_length = 512
    n_fft = 2048
    # data, sampling_rate = librosa.load(audio)
    # chroma_stft = librosa.stft(y = data)
    # print("Chroma stft: ", chroma_stft)
    # chroma_stft_mean = abs(chroma_stft.mean())
    # chroma_stft_var = librosa.stft(y = data).var()

    chroma_stft = np.abs(librosa.stft(y = data, n_fft = n_fft, hop_length = hop_length))
    chroma_stft_mean = chroma_stft.mean()
    chroma_stft_var = chroma_stft.var()

    rms_mean = librosa.feature.rms(y = data)[0].mean()
    rms_var = librosa.feature.rms(y = data)[0].var()

    spectral_centroid_mean = librosa.feature.spectral_centroid(y = data, sr = sampling_rate)[0].mean()
    spectral_centroid_var = librosa.feature.spectral_centroid(y = data, sr = sampling_rate)[0].var()
    
    spectral_bandwidth_mean = librosa.feature.spectral_bandwidth(y = data, sr = sampling_rate)[0].mean()
    spectral_bandwidth_var = librosa.feature.spectral_bandwidth(y = data, sr = sampling_rate)[0].var()
    
    rolloff_mean = librosa.feature.spectral_rolloff(y = data, sr=sampling_rate)[0].mean()
    rolloff_var = librosa.feature.spectral_rolloff(y = data, sr=sampling_rate)[0].var()
    
    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y = data)[0].mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y = data)[0].var()
    
    y_harm, y_perc = librosa.effects.hpss(data)
    harmony_mean = y_harm.mean()
    harmoney_var = y_harm.var()
    perceptr_mean = y_perc.mean()
    perceptr_var = y_perc.var()
    
    tempo, _ = librosa.beat.beat_track(y=data, sr=sampling_rate, units='time')
    
    # X = {
    #     'chroma_stft_mean': chroma_stft_mean,
    #     'chroma_stft_var': chroma_stft_var,
    #     'rms_mean': rms_mean,
    #     'rms_var': rms_var,
    #     'spectral_centroid_mean': spectral_centroid_mean,
    #     'spectral_centroid_var': spectral_centroid_var,
    #     'spectral_bandwidth_mean': spectral_bandwidth_mean,
    #     'spectral_bandwidth_var': spectral_bandwidth_var,
    #     'rolloff_mean': rolloff_mean,
    #     'rolloff_var': rolloff_var,
    #     'zero_crossing_rate_mean': zero_crossing_rate_mean,
    #     'zero_crossing_rate_var': zero_crossing_rate_var,
    #     'harmony_mean': harmony_mean,
    #     'harmony_var': harmoney_var,
    #     'perceptr_mean': perceptr_mean,
    #     'perceptr_var': perceptr_var,
    #     'tempo': tempo
    # }

    # X_df = pd.Series(X)
    X_df = np.array([chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean, spectral_centroid_var,
         spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean, rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var,
         harmony_mean, harmoney_var, perceptr_mean, perceptr_var, tempo])
    return X_df

def find_genre(X):
    X_2d = X.reshape(1, -1)
    
    # print("X_2d type: ", type(X_2d))
    # print("X_2d: ", X_2d)
    with open('scaler.pickle', 'rb') as file:
        sc = pickle.load(file)

    # X_scaled = sc.fit_transform(X_2d)
    # print(X_scaled)
    X_scaled = sc.transform(X_2d)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # print("X_tensor: ", X_tensor)
    # X_tensor = torch.tensor(X, dtype=torch.float32)
    model = pickle.load(open('model.sav', 'rb'))
    outputs = model(X_tensor.float())
    # print(outputs)
    _, predicted = torch.max(outputs, 1)
    # print("Checking: ", torch.max(outputs, 1))
    
    genre_dict = {
        0: 'blues',
        1: 'classical',
        2: 'country',
        3: 'disco',
        4: 'hiphop',
        5: 'jazz',
        6: 'metal',
        7: 'pop',
        8: 'reggae',
        9: 'rock',
        }
    
    predicted =  predicted.tolist()[0]
    return genre_dict[predicted]

def split_extract_predict(input_file, duration):
    
    # Load the input .wav file
    audio, sr = librosa.load(input_file)

    # Calculate the number of samples in the desired duration
    samples_per_segment = int(duration * sr)

    # Calculate the number of segments
    num_segments = len(audio) // samples_per_segment

    # Split the file into segments

    pred = []
    for i in range(num_segments):
        # Get the start and end sample indices for the segment
        start = i * samples_per_segment
        end = start + samples_per_segment

        # Extract the segment from the audio
        segment = audio[start:end]

        #Create features from the segment
        features = extract_features(segment, sr)
        
        pred.append(find_genre(features))

    return pred

audio_file_path = sys.argv[1]

with open('scaler.pickle', 'rb') as file:
    sc = pickle.load(file)
duration = 3
X = split_extract_predict(audio_file_path, duration)
genre = mode(X)

# print("list: ", X)
print("The python script only computes for 10 genres which are 'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock',")
print("From the list, the genre of ", f"{sys.argv[1]}", "is computed as : ", genre)