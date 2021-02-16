#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
from glob import glob
import itertools
import random
import matplotlib.pylab as plt
import sklearn
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


imagePatches = glob('../input/breast-histopathology-images/IDC_regular_ps50_idx5/**/*.png', recursive=True)
for filename in imagePatches[0:10]:
    print(filename)


# In[5]:


# Two arrays holding images by class type

class0 = [] # 0 = no cancer
class1 = [] # 1 = cancer

for filename in imagePatches:
    if filename.endswith("class0.png"):
         class0.append(filename)
    else:
        class1.append(filename)


# In[6]:


sampled_class0 = random.sample(class0, 30000)
sampled_class1 = random.sample(class1, 30000)
len(sampled_class0)


# In[7]:


from matplotlib.image import imread
import cv2

def get_image_arrays(data, label):
    img_arrays = []
    for i in data:
      if i.endswith('.png'):
        img = cv2.imread(i ,cv2.IMREAD_COLOR)
        img_sized = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
        img_arrays.append([img_sized, label])
    return img_arrays


# In[8]:


class0_array = get_image_arrays(sampled_class0, 0)
class1_array = get_image_arrays(sampled_class1, 1)


# In[9]:


test = cv2.imread('../input/breast-histopathology-images/IDC_regular_ps50_idx5/13689/1/13689_idx5_x801_y1501_class1.png' ,cv2.IMREAD_COLOR)
test.shape


# In[10]:


combined_data = np.concatenate((class0_array, class1_array))
random.shuffle(combined_data)


# In[11]:


X = []
y = []

for features,label in combined_data:
    X.append(features)
    y.append(label)


# In[12]:


# print(X[11].reshape(-1, 50, 50, 3))
# reshape X data
X = np.array(X).reshape(-1, 50, 50, 3)


# In[14]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[15]:


X_train.shape


# In[16]:


v19_model = keras.applications.vgg19.VGG19(weights=None, input_shape=(50,50,3), include_top=False)
for new_layer, layer in zip(v19_model.layers[1:], v19_model.layers[1:]):
    new_layer.set_weights(layer.get_weights())

for layer in v19_model.layers: 
    layer.trainable = False  
    
    
    
model_test_vgg = Sequential()

for layer in v19_model.layers:
    model_test_vgg.add(layer)
    
model_test_vgg.add(Flatten()) 
model_test_vgg.add(Dense(2, activation='softmax', name='Predictions')) 
model_test_vgg.layers[-2].trainable = True 
model_test_vgg.layers[-3].trainable = True  
model_test_vgg.layers[-4].trainable = True  
model_test_vgg.layers[-5].trainable = True  
model_test_vgg.layers[-6].trainable = True 


# In[ ]:


model_test_vgg.compile(optimizer=RMSprop(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model_test_vgg.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=150)


# In[ ]:


model_test_vgg.summary()


# In[ ]:





# In[ ]:





# In[ ]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

