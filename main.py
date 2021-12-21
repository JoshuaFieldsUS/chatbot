import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

##CODE IS UNUSABLE AT THE MOMENT | RUN CHATBOT.PY FOR CURRENT BUILD

img_array = cv2.imread('train/0/Training_3908.jpg')
print(img_array.shape)
plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
plt.show()
Datadirectory = "train/"
Classes = ["0","1","2","3","4","5","6"]

for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        #backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break

img_size = 224
new_array = cv2.resize(img_array, (img_size,img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()
print(new_array.shape)

# Read all the images and convert them into an array

training_Data = [] # Data Array

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array, (img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_Data()
print(len(training_Data))

#temp = np.array(training_Data)
#print(temp.shape)

import random
random.shuffle(training_Data)

X = []  # Data / Feature
y = []  # label

for features,label in training_Data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, img_size, img_size, 3) # Converting it to 4 dimension

print(X.shape)
# Normalize the data
X = X/255.0
Y = np.array(y)
print(Y.Shape)

# Deep learning model for training - Transfer Learning

model = tf.keras.applications.MobileNetV2() ##Pre_Trained Model

print('NOTHING CRASHED')

