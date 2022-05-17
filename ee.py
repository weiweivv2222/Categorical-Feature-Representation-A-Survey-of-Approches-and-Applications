# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:01:55 2021

@author: xg16060
"""

import numpy as np 
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import tensorflow as tf
import random as rn

#%% keep the result can be reproduce
import os
os.environ['PYTHONHASHSEED']='0'

np.random.seed(42)
rn.seed(12345)

session_conf=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)

from keras import backend 

tf.random.set_seed(1234)
sess=tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
#build the embedding layers
model=Sequential()
model.add(Embedding(10,3,input_length=2))

input_array=np.random.randint(10,size=(32,2))

#model.compile(optimizer='rmsprop','mse')
model.compile('rmsprop','mse')

output_array=model.predict(input_array)

weight=model.get_weights()
