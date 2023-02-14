
import pickle
import random
import itertools
import classifier

from scipy import misc
from six.moves import xrange
from IPython.display import HTML
from moviepy.editor import VideoFileClip

from os import listdir
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf

from PIL import Image
import os
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Rescaling, AveragePooling2D, Dropout,MaxPool2D

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score

data = []
labels = []
classes = 16
SIZE = 32
cur_path = os.getcwd()

#Retrieving the images and their labels 
for i in range(classes):
    path = os.path.join(cur_path,'Train',str(i))
    images = os.listdir(path)

    for j in images:
            try:
                image = Image.open(path + '\\'+ j).convert('L')
                image = image.resize((SIZE,SIZE))
                image = np.array(image)
                #sim = Image.fromarray(image)
                data.append(image)
                label = np.zeros(classes)
                label[i] = 1.0
                labels.append(label)
               
            except:
                print("Error loading image")
                
                
#Converting lists into numpy arrays
data = np.array(data)
data = data/255
labels = np.array(labels)

print('Images shape:', data.shape)
print('Labels shape:', labels.shape)

X = data.astype(np.float32)
y = labels.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

X_train = tf.expand_dims(X_train, axis=-1)


SignName=pd.read_csv("Sign_Names.csv")

SignNames=pd.Series(SignName.SignName.values,index=SignName.ClassId).to_dict()



plt.figure(figsize=(18, 30))
start_index = 0
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    label = np.argmax(y_train[start_index+i])
    sign = SignNames[label]
    
    plt.xlabel((sign))
    plt.imshow(X_train[start_index+i], cmap='gray')
plt.show()



# Building the model
model = Sequential([
    Rescaling(1, input_shape=(32, 32, 1)),
    Conv2D(filters=6, kernel_size=(5, 5), activation='relu'),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(filters=120, kernel_size=(5, 5), activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(units=120, activation='relu'),
    Dense(units=16, activation='softmax')
])

# Compilation of the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model architecture
model.summary()


history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

val_loss, val_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nValidation accuracy:', val_acc)
print('\nValidation loss:', val_loss)


plt.figure(0)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')

plt.figure(1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 0.2])
plt.legend(loc='lower right')


preds = model.predict(X_test)

plt.figure(figsize=(18, 30))
start_index = random.randint(0, 1500)
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    pred = np.argmax(preds[start_index+i])
    gt = np.argmax(y_test[start_index+i])
    sign = SignNames[gt]
    
    col = 'g'
    if pred != gt:
        col = 'r'
    
    plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i, pred, sign), color=col)
    plt.imshow(X_test[start_index+i], cmap='gray')
plt.show()

model.save("my_model")


lb = LabelBinarizer()

classes = SignName.iloc[:,1].values
labels = lb.fit_transform(labels)

f=open("label_bin.pickle","wb")

f.write(pickle.dumps(lb))
f.close()