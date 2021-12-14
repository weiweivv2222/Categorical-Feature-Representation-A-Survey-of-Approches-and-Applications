import pandas as pd
import numpy as np
from numpy import interp
import sys
import category_encoders as ce

import numpy as np 
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import tensorflow as tf
import random as rn
import pandas as pd
import os
from keras import backend 
#%%#%% Entity embedding approaches 
#keep the result can be reproduce
os.environ['PYTHONHASHSEED']='0'
np.random.seed(42)
rn.seed(12345)
session_conf=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
tf.random.set_seed(1234)
sess=tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

#build the embedding layers
model=Sequential()
model.add(Embedding(12,5,input_length=1))
model.compile('Adam','mape')

#convert job feature with one-hot encoding by indices of the featuer 
label_encoder=ce.OrdinalEncoder(cols=['Job']) 
input_array=label_encoder.fit_transform(df.Job)
input_array=input_array-1

# unique_category_count = 12
# inputs = tf.one_hot(input_array, unique_category_count)
# output_array_o=model.predict(inputs)

output_array=model.predict(input_array)
weight=model.get_weights()
#%%
import tensorflow as tf

category_indices = [0, 1, 2, 2, 1, 0]
unique_category_count = 3
inputs = tf.one_hot(category_indices, unique_category_count)
print(inputs.numpy())