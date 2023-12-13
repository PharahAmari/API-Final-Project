import json
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# from scikitplot.metrics import plot_roc

from datetime import datetime

import lazy_loader as lazy
resampy = lazy.load("resampy")


            #####################################################
            ### -------------- MODEL STRUCTURE -------------- ###
            #####################################################

def create_model(num_labels):
    model = Sequential()

    # First layer
    model.add(Dense(100, input_shape=(40,)))
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

# Change according to your system
DATASET_PATH = 'dataset_api.json'
TEST_DATASET_PATH = 'test_dataset.json'


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
def plot_classes(class_data):
    plt.figure(figsize=(15, 8))
    plt.title('Alarm Sounds Class Distribution', fontsize=22)
    class_distribution = class_data.value_counts().sort_values()
    palette = sns.color_palette("husl", len(class_distribution.values))
    sns.barplot(x=class_distribution.values,
                y=list(class_distribution.keys()),
                orient="h", palette=palette)

    plt.show()
def plot_confusion_matrix(y_true, y_pred, classes='auto', figsize=(10, 10), text_size=12): 
    # Generate confusion matrix 
    cm = confusion_matrix(y_true, y_pred)
    
    # Set plot size
    plt.figure(figsize=figsize)

    # Create confusion matrix heatmap
    disp = sns.heatmap(
        cm, annot=True, cmap='Greens',
        annot_kws={"size": text_size}, fmt='g',
        linewidths=1, linecolor='black', clip_on=False,
        xticklabels=classes, yticklabels=classes)
    
    # Set title and axis labels
    disp.set_title('Confusion Matrix', fontsize=24)
    disp.set_xlabel('Predicted Label', fontsize=20) 
    disp.set_ylabel('True Label', fontsize=20)
    plt.yticks(rotation=0) 

    # Plot confusion matrix
    plt.show()

            #####################################################
            ### ----------------- LOAD DATA ----------------- ###
            #####################################################

with open(DATASET_PATH, 'r') as file:
    # Load JSON data from file
    data = json.load(file)

label_list, file_list = extract_data(data)
print('System: Data loaded.')

extracted_features = []
for i in range(len(file_list)):
    final_class_labels = label_list[i]
    data = features_extractor("." + file_list[i])
    extracted_features.append([data, final_class_labels])

extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())
plot_classes(extracted_features_df['class'])
print('System: Extracted features.')

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

num_labels = y.shape[1]
print('System: Generated datasets for training.')


            #####################################################
            ### ------------------- TRAIN ------------------- ###
            #####################################################

# Hyperparameters
num_epochs = 50
num_batch_size = 4

# Create the model and print its structure
model = create_model(num_labels)
print('System: Model created.')
print(model.summary())

# Save model config
checkpointer = ModelCheckpoint(filepath=f'saved_models/audio_class_ep{num_epochs}_bs{num_batch_size}.hdf5',
                               verbose=1, save_best_only=True)
# Train
start = datetime.now()
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test),
          callbacks=[checkpointer], verbose=1)
duration = datetime.now() - start
print("System: Training completed in time: ", duration)

# Save model training
model.save(f'saved_models/audio_class_trained_ep{num_epochs}_bs{num_batch_size}.h5')

# Plot confusion matrix
predictions = model.predict(X_test)
pred_classes = np.argmax(predictions, axis = 1)
plot_confusion_matrix(np.argmax(y_test, axis=1), pred_classes, classes=labelencoder.classes_)
