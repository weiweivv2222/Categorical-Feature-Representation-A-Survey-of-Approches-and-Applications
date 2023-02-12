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
# df=df_bank

# cat_cols=cat_cols_bank
df = df_adult
cat_cols=cat_cols_adult
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
#%% RFNE

start_time = time.time()

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import *
from sklearn.tree import DecisionTreeClassifier
# import labelencoder
from sklearn.preprocessing import LabelEncoder, StandardScaler# instantiate labelencoder object
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier

y =  df['y']
X = df.drop(['y'], axis=1)

#label encoding.

X = pd.get_dummies(X, prefix_sep='_', drop_first=True) #One hot encoding, assuming complet knowledge of vocabulary

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0 )

scl = StandardScaler()

X_train = scl.fit_transform(X_train)
X_test = scl.transform(X_test)


estimator_list = []
estimators = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
estimators.fit(X_train,y_train)
estimator = estimators.estimators_[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = seed,stratify=y)

# The decision estimator has an attribute called tree_  which stores the entire
# tree structure and allows access to low level attributes. The binary tree
# tree_ is represented as a number of parallel arrays. The i-th element of each
# array holds information about the node `i`. Node 0 is the tree's root. NOTE:
# Some of the arrays only apply to either leaves or split nodes, resp. In this
# case the values of nodes of the other type are arbitrary!
#
# Among those arrays, we have:
#   - left_child, id of the left child of the node
#   - right_child, id of the right child of the node
#   - feature, feature used for splitting the node
#   - threshold, threshold value at the node
#

# Using those arrays, we can parse the tree structure:

n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold



# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 X.columns[feature[i]],
                 threshold[i],
                 children_right[i],
                 ))
print()


# First let's retrieve the decision path of each sample. The decision_path
# method allows to retrieve the node indicator functions. A non zero element of
# indicator matrix at the position (i, j) indicates that the sample i goes
# through the node j.

node_indicator = estimator.decision_path(X_test)

# Similarly, we can also have the leaves ids reached by each sample.

leave_id = estimator.apply(X_test)

# Now, it's possible to get the tests that were used to predict a sample or
# a group of samples. First, let's make it for the sample.

sample_id = 0
node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

# create graphs, both tress and leaves graph
import networkx as nx

def create_graph_list(estimators_trees):
    G_list = []
    G_list_leaves =[]
    
    for estimator in estimators_trees.estimators_ :
        G=nx.Graph() #
        G_leaves = nx.Graph() # leaves graph

        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1,0)]  # seed is the root node id and its parent depth

        while len(stack) > 0:
            node_id, parent_depth,parent_id = stack.pop()
            node_depth[node_id] = parent_depth + 1
            G.add_node(str(node_id))
            G.add_edge(str(parent_id),str(node_id))
            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
        
                stack.append((children_left[node_id], parent_depth + 1,node_id))
                stack.append((children_right[node_id], parent_depth + 1,node_id))
            else:
                is_leaves[node_id] = True
        #append created graph
        
        for i in range(0,len(is_leaves)):
            if (is_leaves[i]):
                G_leaves.add_node(str(i))
        
                for j in range(0,len(is_leaves)):
                       if (is_leaves[j]):         
                           if i!=j:
                                G_leaves.add_node(str(j))
                                G_leaves.add_edge(str(i),str(j),weight=nx.shortest_path_length(G,source=str(i),target=str(j)))
        
        G_list.append(G)
        G_list_leaves.append(G_leaves)
        #return multi graph
    return G_list,G_list_leaves

# # Create Trees and Leaves Graphs
#The leaves graph connects each of the leaves taking the hops between leaves as the distance.
import matplotlib.pyplot as plt
from node2vec import Node2Vec

G_list_leaves,G_list = create_graph_list(estimators)

G = G_list[0]
nx.draw(G,with_labels = True)
#plt.savefig('mytree_many.png',format='png', dpi=1200)

#plt.show()

A = nx.adjacency_matrix(G)


#plt.imshow(A.todense(), cmap='hot', interpolation='nearest')

#plt.show()

np.shape(A.todense())

from tqdm import tqdm_notebook as tqdm

node2vec_list = []
for G in tqdm(G_list):
    node2vec = Node2Vec(G, dimensions=10, walk_length=15, num_walks=100)
    node2vec_list.append(node2vec)
    
    
# Learn embeddings # Learn embeddings     
models_list = []

for node2v in node2vec_list:
    model = node2v.fit(window=5, min_count=1)
    models_list.append(model)

# plot the first community
to_plot = []

model = models_list[0]

for i in model.wv. index_to_key:
    to_plot.append(model.wv.get_vector(i))


ar = np.array(to_plot)
plt.plot(ar[:,0], ar[:,1], 'o',  linewidth=2, markersize=20)

index =0 
for i in model.wv. index_to_key:
    plt.text(ar[index,0], ar[index,1], i)
    index+=1

plt.savefig('embedding_result_many.png',format='png', dpi=1200)
plt.show()
    
# extract embeddings on Data
reps = []
reps_test = []
for estimator, model in tqdm(zip(estimators.estimators_, models_list)):
    leave_id = estimator.apply(X_train)
    vect_rep = [model.wv.get_vector(str(i)) for i in leave_id if str(i) in model.wv.index_to_key]
    leave_id_test = estimator.apply(X_test)
    vect_rep_test = [model.wv.get_vector(str(i)) for i in leave_id_test if str(i) in model.wv.index_to_key]
    if reps == []:
        reps.append(vect_rep)
        reps_test.append(vect_rep_test)
    else:
        reps = np.array(reps).squeeze()
        reps_test = np.array(reps_test).squeeze()
        reps = np.hstack((reps, np.array(vect_rep)))
        reps_test = np.hstack((reps_test, np.array(vect_rep_test)))
        print(np.shape(reps))
data = np.array(reps)
data_test = np.array(reps_test)

# concatenate the embedidng data with the original features with One-hot encoding 
X_train_embeded= data #np.hstack((X_train,data))
X_test_embeded= data_test # np.hstack((X_test,data_test))

X_train_embeded= pd.DataFrame(np.hstack((X_train,data)))
X_test_embeded= pd.DataFrame(np.hstack((X_test,data_test)))


print('computation time of RFNE :',time.time() - start_time)
print('Memory usage of x train data after encoding: ',round(X_train_embeded.memory_usage(deep=True).sum()* BYTES_TO_MB,3))
print('Memory usage of x  test after encoding: ',round(X_test_embeded.memory_usage(deep=True).sum()* BYTES_TO_MB,3))
print('Memory usage of y train after encoding: ',round(y_train.memory_usage(deep=True)* BYTES_TO_MB,3))
print('Memory usage of y test after encoding: ',round(y_test.memory_usage(deep=True)* BYTES_TO_MB,3))

#%% parameters for RFNE
X_train = X_train_embeded
X_test = X_test_embeded
num_fold = 5
number_of_features= X.shape[1]
skf = StratifiedKFold(n_splits=num_fold,random_state=seed, shuffle = True)
#%% Train-Test split for discrete function approach
num_fold = 5
X=woe_encoder_transformed.drop(['y'], axis=1)
#X = df.drop(['y'], axis=1)
#y=df['y']
y =  df['y']
number_of_features=  X.shape[1]
skf = StratifiedKFold(n_splits=num_fold,random_state=seed, shuffle = True)


    
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

