import json
import librosa
import numpy as np
from tqdm import tqdm
import os
import lazy_loader as lazy
resampy = lazy.load("resampy")
import pandas as pd

with open('dataset_api.json', 'r') as file:
    # 从文件中加载JSON数据
    data = json.load(file)

api_data = data['api_dataset']
def extract_data(api_data):
    label_list = []
    file_list = []
    for i in range(len(api_data)):
        for file in api_data[i]['files']:
            label_list.append(api_data[i]['class'])
            file_list.append(file)
    return label_list, file_list
label_list, file_list = extract_data(api_data)


def features_extractor(file):

    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features
extracted_features=[]
for i in range(len(file_list)):

    final_class_labels = label_list[i]
    data = features_extractor("." + file_list[i])
    extracted_features.append([data, final_class_labels])

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
num_labels=y.shape[1]
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 100
num_batch_size = 2

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)