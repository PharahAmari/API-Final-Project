import os
import json
import pickle
import librosa
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datetime import datetime

import lazy_loader as lazy
resampy = lazy.load("resampy")


###########################################################
### ----------------- MODEL STRUCTURE ----------------- ###
###########################################################

def create_model():
    model=Sequential()

    # First layer
    model.add(Dense(100,input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Second layer
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Third layer
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Final layer
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adam')

    return model

#####################################################
### ----------------- CONSTANTS ----------------- ###
#####################################################

# change according to your system
DATASET_PATH = 'C:/Users/wilru/Documents/LU/S3/API/P1/API-Final-Project/dataset_api.json'

#####################################################
### ----------------- FUNCTIONS ----------------- ###
#####################################################

def extract_data(data):
    label_list = []
    file_list = []
    for i in range(len(data)):
        for file in data[i]['files']:
            label_list.append(data[i]['class'])
            file_list.append(file)
    return label_list, file_list

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features


#####################################################
### ----------------- LOAD DATA ----------------- ###
#####################################################

with open(DATASET_PATH, 'r') as file:
    # 从文件中加载JSON数据
    data = json.load(file)

label_list, file_list = extract_data(data)
print('System: Data loaded,')

extracted_features=[]
for i in range(len(file_list)):

    final_class_labels = label_list[i]
    data = features_extractor("." + file_list[i])
    extracted_features.append([data, final_class_labels])

extracted_features_df = pd.DataFrame(extracted_features,columns=['feature','class'])
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

print('System: Extracted features.')

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

num_labels=y.shape[1]
print('System: Generated datasets for trianing.')

##########################################################
### ----------------- TRAINING MODEL ----------------- ###
##########################################################

# Hyperparameters
num_epochs = 100
num_batch_size = 2

# Save model config
checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5',
                               verbose=1, save_best_only=True)

# Create the model and print its structure
model = create_model()
print('System: Model created.')
print(model.summary())

# Train
start = datetime.now()
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
duration = datetime.now() - start
print("System: Training completed in time: ", duration)

# Save model training
with open('./models/model_v01.pkl', 'wb') as f:
    pickle.dump(model, f)

# # Load model
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

