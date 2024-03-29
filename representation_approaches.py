

import pandas as pd
import numpy as np
from numpy import interp
import sys
import category_encoders as ce

#%%
'''load the data'''
data_path='C:/Users/xg16060/OneDrive - APG/PhD/3_categorical_embedding_3rd/data/'
bank_data = pd.read_csv(data_path+'bank-additional-full.csv', sep=';')

seed = 7
np.random.seed(seed)
#%%#%data pre-processing 
bank_data.isnull().sum()
bank_data.dropna(inplace=True)
# set up visualization 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#bank_data.describe()
#bank_data.info()

bank_data = bank_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
bank_data.replace('?',np.nan,inplace=True)

bank_data.dropna(inplace=True)
bank_data= bank_data[(bank_data=='?')==False]

# Get all categorical features
cat_columns = list(bank_data.columns[bank_data.dtypes=='object'])

bank_data['y'] = (bank_data['y']=='yes').apply(int) 

#%% an example categorical feature: job 
job=bank_data[['job','y']]

#the frequency table of the feature Job
job['job'].value_counts()

job.columns=['Job','Target']
#random select the 10 samples from the dataset
df=job.sample(n=20, random_state=12,replace=True)

#shuffer the rows 
df=job.sample(frac=1, random_state=12)
#%% different embedding 
# one-hot encoding 
one_hot_encoder=ce.OneHotEncoder(cols=['Job']) 
df_one_hot_transformed=one_hot_encoder.fit_transform(df)
print(df_one_hot_transformed.iloc[0:7,])

# %%label encode
label_encoder=ce.OrdinalEncoder(cols=['Job']) 
df_label_transformed=label_encoder.fit_transform(df)
print(df_label_transformed.iloc[0:7,])

#%%hash encoding  with md5 hash function

hash_encoder=ce.HashingEncoder(cols=['Job'],n_components=7)
hash_transformed=hash_encoder.fit_transform(df)
print(hash_transformed.iloc[0:7,])


#%%  target encoding
target_encoder=ce.TargetEncoder(cols='Job',smoothing=1)
mean_target_transformed=target_encoder.fit_transform(df['Job'],df['Target'])
print(mean_target_transformed.iloc[0:7,])

#%%WoE
woe_encoder=ce.WOEEncoder(cols='Job')
woe_encoder_transformed=woe_encoder.fit_transform(df['Job'],df['Target'])
print(woe_encoder_transformed.iloc[0:7,])
y=df[df['Job']=='student']


#%% Entity embedding approaches 
#keep the result can be reproduce
import numpy as np 
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import tensorflow as tf
import random as rn
import pandas as pd
import os
from keras import backend 
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

import tensorflow as tf
category_indices = [0, 1, 2, 2, 1, 0]
unique_category_count = 3
inputs = tf.one_hot(category_indices, unique_category_count)
print(inputs.numpy())
