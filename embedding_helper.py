import pandas as pd
import os
import numpy as np
from numpy import interp
import sys
import random
import tensorflow as tf
seed=123
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

import category_encoders as ce
from keras.layers.embeddings import Embedding
from keras.models import Sequential

from keras import backend 
from keras import models
from keras import layers


#%% create a NN model
# Create function returning a compiled network
def create_network(number_of_features):
    
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=32, activation='relu', input_shape=(number_of_features,)))

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=32, activation='relu'))
    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=16, activation='relu'))
    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # Compile neural network
    network.compile(loss='binary_crossentropy', # Cross-entropy
                    optimizer='rmsprop', # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric
    
    # Return compiled network
    return network
#%%
# #%% Entity embedding approaches 
# #keep the result can be reproduce
# os.environ['PYTHONHASHSEED']='0'
# np.random.seed(42)
# rn.seed(12345)
# session_conf=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
#                             inter_op_parallelism_threads=1)
# tf.random.set_seed(1234)
# sess=tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)

# #build the embedding layers
# model=Sequential()
# model.add(Embedding(12,5,input_length=1))
# model.compile('Adam','mape')

# #convert job feature with one-hot encoding by indices of the featuer 
# label_encoder=ce.OrdinalEncoder(cols=['Job']) 
# input_array=label_encoder.fit_transform(df.Job)
# input_array=input_array-1

# # unique_category_count = 12
# # inputs = tf.one_hot(input_array, unique_category_count)
# # output_array_o=model.predict(inputs)

# output_array=model.predict(input_array)
# weight=model.get_weights()
# #%%
# import tensorflow as tf

# category_indices = [0, 1, 2, 2, 1, 0]
# unique_category_count = 3
# inputs = tf.one_hot(category_indices, unique_category_count)
# print(inputs.numpy())