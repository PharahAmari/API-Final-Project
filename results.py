import json
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import lazy_loader as lazy
resampy = lazy.load("resampy")


            #####################################################
            ### ----------------- CONSTANTS ----------------- ###
            #####################################################

# Change to load a different version
MODEL_PATH = 'saved_models/audio_classification.hdf5'
TEST_DATASET_PATH = 'C:/Users/wilru/Documents/LU/S3/API/P1/API-Final-Project/test_dataset.json'

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

def plot_roc_curve(y_true, y_probabilities, classes, class_name):
    # Binarize the labels for each class
    y_true_bin = label_binarize(y_true, classes=classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f'{class_name[i]} (AUC = {roc_auc[i]:.2f})')

    # Plot random guessing line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

    # Set plot details
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-Class Classification')
    plt.legend(loc='lower right')
    plt.show()

def report(test_labels_encoded, test_predictions, class_names):
    print(classification_report(
        test_labels_encoded, 
        test_predictions, 
        target_names=class_names
    ))

            #####################################################
            ### ------------ LOAD MODEL AND DATA ------------ ###
            #####################################################

# Evaluate the model on the test data
with open(TEST_DATASET_PATH, 'r') as test_file:
    test_data = json.load(test_file)

# Load test data
test_label_list, test_file_list = extract_data(test_data)

# Extract features
test_extracted_features = []
for i in range(len(test_file_list)):
    final_class_labels = test_label_list[i]
    test_data = features_extractor("." + test_file_list[i])
    test_extracted_features.append([test_data, final_class_labels])

# Add data to correct format
test_extracted_features_df = pd.DataFrame(test_extracted_features, columns=['feature', 'class'])
X_test = np.array(test_extracted_features_df['feature'].tolist())
y_test = np.array(test_extracted_features_df['class'].tolist())

# Encode (one-hot)
labelencoder = LabelEncoder()
y_test = to_categorical(labelencoder.fit_transform(y_test))

# Load the trained model
loaded_model = load_model(MODEL_PATH)


            #####################################################
            ### ------------------ TESTING ------------------ ###
            #####################################################

# Evaluate the model on the test data
results = loaded_model.evaluate(X_test, y_test)
print("Test loss:", results[0])
print("Test accuracy:", results[1])

# Predictions on the test data
predictions = loaded_model.predict(X_test)
pred_classes = np.argmax(predictions, axis = 1)

# Plot confusion matrix
plot_confusion_matrix(np.argmax(y_test, axis=1), pred_classes, classes=labelencoder.classes_)

# Plot ROC Curve
test_labels_encoded = np.argmax(y_test, axis=1)
unique_classes = np.unique(test_labels_encoded)

# Plot ROC curves
class_name = ['alarms', 'doorbell', 'glass breacking', 'gunshot', 'routine', 'screaming']
plot_roc_curve(test_labels_encoded, predictions, unique_classes, class_name)
