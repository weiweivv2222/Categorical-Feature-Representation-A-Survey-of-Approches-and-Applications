import pandas as pd
import numpy as np
from numpy import interp
#%% bank data
def bank_data_prep(bank_data):
    '''
    input: data frame. here the data propressing is customized for bank data
    '''
    
    #check the NA values in bankdata
    #print('check the NAs in bank_data', bank_data.isnull().sum())
        
    #remove useless columns
    re_cols=['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m','nr.employed']
    bank_data.drop(re_cols,axis=1,inplace=True)
    # set up visualization 
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    #change letters to strings
    bank_data = bank_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    #replace "?" by na
    bank_data.replace('?',np.nan,inplace=True)
    
    #remove NAs
    bank_data.dropna(inplace=True)
    
    # Get all categorical features
    cat_columns = list(bank_data.columns[bank_data.dtypes=='object'])
    cat_columns.remove('y')
    #dummy target variable
    bank_data['y'] = (bank_data['y']=='yes').apply(int) 
    
   
    #bank_input=bank_data.drop(columns=['y'],axis=1)

    #bank_data.drop(columns=['age'],axis=1, inplace=True)
    return(bank_data,cat_columns)
#%%adult 
def adult_data_prep(adult_data):
    '''
    input: data frame. here the data propressing is customized for bank data
    '''
    #check the NA values in bankdata
    #print('check the NAs in adult_data', adult_data.isnull().sum())
      
   # adult_data.drop(re_cols,axis=1,inplace=True)
    # set up visualization 
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    #change letters to strings
    adult_data = adult_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    #adult_data.info()
    #adult_data.describe()
    #replace "?" by na
    adult_data.isin(['?']).sum()
    adult_data.replace('?',np.nan,inplace=True)
    
    #remove NAs
    adult_data.dropna(inplace=True)
    
    # Get all categorical features
    cat_columns = list(adult_data.columns[adult_data.dtypes=='object'])
    cat_columns.remove('income')
    #dummy target variable
    adult_data['income'] =  adult_data['income'].map({'<=50K': 0, '>50K': 1})
    
   
    #bank_input=adult_data.drop(columns=['y'],axis=1)

    #adult_data.drop(columns=['age'],axis=1, inplace=True)
    return(adult_data,cat_columns)

