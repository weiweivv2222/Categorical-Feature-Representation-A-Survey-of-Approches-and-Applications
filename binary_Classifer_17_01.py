import config_cat_embedding
import pandas as pd
import numpy as np
import time
import os
import random
import tensorflow as tf
seed=123
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split,cross_val_score,GridSearchCV,cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score ,roc_auc_score,precision_recall_curve,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline


import warnings
# "error", "ignore", "always", "default", "module" or "once"
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from data_prep import bank_data_prep,adult_data_prep
from multiscorer import MultiScorer
from embedding_helper import create_network
#%% load the data and completed the data pre-processing

data_path = config_cat_embedding.paths['data']
data_path_out = config_cat_embedding.paths['data_output']
bank_data = pd.read_csv(data_path+'bank-additional-full.csv', sep=';')
adult_data=pd.read_csv(data_path+'adult.csv', sep=',')

#data pre-processing 
df_bank, cat_cols_bank=bank_data_prep(bank_data)
df_adult, cat_cols_adult=adult_data_prep(adult_data)
df=df_bank

cat_cols=cat_cols_bank
#df = df_adult
#cat_cols=cat_cols_adult
#%%calculate the memory usage of the prepared data frame
BYTES_TO_MB = 0.000001

print(round(df.memory_usage(deep=True).sum()* BYTES_TO_MB, 3))

#round(df_adult.memory_usage(deep=True).sum()* BYTES_TO_MB, 3)

#adult_data.info(memory_usage='deep')
#%% different embedding 
# one-hot encoding 
start_time = time.time()
one_hot_encoder=ce.OneHotEncoder(cols=cat_cols) 
one_hot_transformed=one_hot_encoder.fit_transform(df)
print('computation time of one-hot :',time.time() - start_time)
print('Memory usage after encoding: ',round(one_hot_transformed.memory_usage(deep=True).sum()* BYTES_TO_MB,3))

# label encode
start_time = time.time()
label_encoder=ce.OrdinalEncoder(cols=cat_cols) 
label_transformed=label_encoder.fit_transform(df)
print('computation time of label:',time.time() - start_time)
print('Memory usage after encoding: ',round(label_transformed.memory_usage(deep=True).sum()* BYTES_TO_MB,3)) 

#hash encoding  with md5 hash function: bank data with 9, adult data=5
start_time = time.time()
hash_encoder=ce.HashingEncoder(cols=cat_cols,n_components=9)
hash_transformed=hash_encoder.fit_transform(df)
print('computation time of hash:',time.time() - start_time)
print('Memory usage after encoding: ',round(hash_transformed.memory_usage(deep=True).sum()* BYTES_TO_MB,3))


#target encoding 
start_time = time.time()
target_encoder=ce.TargetEncoder(cols=cat_cols,smoothing=1)
#mean_target_transformed=target_encoder.fit_transform(df[cat_cols],df['y'])
mean_target_transformed=target_encoder.fit_transform(df,df['y'])

print('computation time of target:',time.time() - start_time)
print('Memory usage after encoding: ',round(mean_target_transformed.memory_usage(deep=True).sum()* BYTES_TO_MB,3))

#WoE
start_time = time.time()
woe_encoder=ce.WOEEncoder(cols=cat_cols)
woe_encoder_transformed=woe_encoder.fit_transform(df,df['y'])
print('computation time of WOE :',time.time() - start_time)
print('Memory usage after encoding: ',round(woe_encoder_transformed.memory_usage(deep=True).sum()* BYTES_TO_MB,3))

# embeddings = [('one hot encoding',df_one_hot_transformed), ('label encoding',df_label_transformed),
#              ('hash encoding',hash_transformed), ('target encoding',mean_target_transformed), ('WOE encoding',woe_encoder_transformed)]

#%% word2vec
start_time = time.time()
from gensim.models import Word2Vec
import numpy as np

# get categorical feature columns

input_data = [df[col].tolist() for col in cat_cols]

# Train the word2vec model
model = Word2Vec(input_data, vector_size=3, window=5, min_count=1)

# Get the vocabulary
vocabulary = model.wv.index_to_key
print("Vocabulary: ", vocabulary)

# Get the vector representation of each word
print("Vector representation of each word:")
for word in vocabulary:
    vector = model.wv.get_vector(word)
    print(word, ":", vector)

# replace the original features by the learned word2Vec embedding

# Loop through all categorical columns
for col in cat_cols:
    # Map each word in the column to its corresponding embedding vector
    df[col + '_embedding'] = df[col].map(lambda x:  model.wv.get_vector(x))

# Drop the original categorical columns
df = df.drop(cat_cols, axis=1)
print('computation time of word2vec :',time.time() - start_time)
print('Memory usage after encoding: ',round(df.memory_usage(deep=True).sum()* BYTES_TO_MB,3))

y =  df['y']

X = df.drop(['y'], axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Get the numerical columns
numerical_cols = X.select_dtypes(exclude='object').columns

# Fit and transform the numerical columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
 
# splite embedding columns 
for col in X.columns:
    if X[col].dtype == object:
        new_cols = [f"{col}_{i}" for i in range(len(X[col][0]))]
        temp = X[col].apply(pd.Series)
        temp.columns = new_cols
        X = pd.concat([X, temp], axis=1)
        X = X.drop([col], axis=1)

#%% Train-Test split for word2vec
num_fold = 5

number_of_features=X.shape[1]
skf = StratifiedKFold(n_splits=num_fold,random_state=seed, shuffle = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = seed,stratify=y)

#%% Train-Test split for discrete function approach
num_fold = 5
X=woe_encoder_transformed.drop(['y'], axis=1)
#X = df.drop(['y'], axis=1)
#y=df['y']
y =  df['y']
number_of_features= 1 #or X.shape[1]
skf = StratifiedKFold(n_splits=num_fold,random_state=seed, shuffle = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = seed,stratify=y)


#%%binary classificatin models
models = [
    ('LR',LogisticRegression(solver= 'liblinear', random_state=seed,max_iter=1000)),
    ('DT',DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=seed)),
    ('RF',RandomForestClassifier(n_estimators=100, max_depth=3, random_state=seed,min_samples_leaf=3)),
    ('KNN',KNeighborsClassifier()),
    ('XGB', XGBClassifier(eval_metric='logloss', use_label_encoder=False)),
    ('SVM',SVC(gamma='scale',random_state=seed)),
    ('MLP', KerasClassifier(build_fn=lambda: create_network(number_of_features), epochs=100, batch_size=100, verbose=0))
    ]
model_names = [model_name[0] for model_name in models]


scorer = MultiScorer({
    'Accuracy'  : (accuracy_score , {}),
    'Precision' : (precision_score, { }),
    'Recall'    : (recall_score   , {}),
    'F1_score'  : (f1_score  , {}),  
    'ROC_AUC' :(roc_auc_score , {})
})


cpt_time=[]
for name, model in models:
    start_time = time.time()
    print (name)
    model_index=model_names.index(name)
    _= cross_val_score(model, X, y, scoring=scorer, cv=skf)
    results = scorer.get_results()

    for metric_name in results.keys():
        average_score = np.average(results[metric_name][num_fold*model_index: num_fold*model_index+num_fold])
        print('%s : %f' % (metric_name, average_score))  
    cmpt_time=time.time() - start_time
    cpt_time.append(cmpt_time)
    print ('Computation time',cmpt_time, '\n\n')
    
#%KPIs in a table
rst=pd.DataFrame(results)
rst_metrics=rst.groupby(np.arange(len(rst))//num_fold).mean()
rst_metrics.index=model_names
rst_metrics['Cpt_time']=cpt_time
rst_metrics=rst_metrics.round(3)
#rst_metrics.sort_index(axis=0,ascending=True,inplace=True)


# #%% an example of LG classifier
# clf=LogisticRegression(random_state=0, max_iter=1000)
# clf.fit(X_train,y_train)
# val_pred=clf.predict(X_test)
# print(f1_score(y_test,val_pred,average='weighted'))
# class_report=classification_report(y_test,val_pred)
# accuracy = model.score(X_test, y_test)

# #save the model to disk
# import pickle
# filename = 'finalized_LG_model.sav'
# pickle.dump(clf, open(filename, 'wb'))

# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# #result = loaded_model.score(X_test, y_test)
# #print(result)

