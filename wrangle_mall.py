#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
from acquire import get_mall_data
import scipy as sp 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import argmax


# Acquire data from mall_customers.customers in mysql database.

# In[20]:


df= get_mall_data()

df.info()


# In[21]:


pd.get_dummies(df.gender, dummy_na=False, drop_first=[True, True])


# In[22]:


def clean_mall(df):
    dummy_df=pd.get_dummies(df.gender, dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_df], axis=1)
    df=df.drop(columns=['Unnamed: 0', 'gender'])
    return df


# In[28]:


df= clean_mall(df)


# Split the data into train, validate, and split

# In[24]:


def split_df(df):
    '''
    This function splits the dataframe in to train, validate, and test.
    '''
    # split dataset
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    return train, validate, test


# In[29]:


train, validate, test = split_df(df)
train.shape, validate.shape, test.shape


# One-hot-encoding (pd.get_dummies)

# Scaling

# In[30]:


def scale_df(train, validate, test):
    '''
    This function scales the split data in the dataframe using the MinMaxScaler and returns the scaled data.
    '''
    # Assign variables
    X_train = train
    X_validate = validate
    X_test = test
    X_train_explore = train

    # Scale data
    scaler = MinMaxScaler(copy=True).fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns.values).set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns= X_validate.columns.values).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled, columns= X_test.columns.values).set_index([X_test.index.values])
    
    return X_train_scaled, X_validate_scaled, X_test_scaled


# In[32]:


X_train_scaled, X_validate_scaled, X_test_scaled = scale_df(train, validate, test)
X_train_scaled.shape, X_validate_scaled.shape, X_test_scaled.shape


# Missing values
