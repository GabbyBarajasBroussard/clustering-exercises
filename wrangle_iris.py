#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import scipy as sp 
import os
import sklearn.preprocessing
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
import acquire


# In[ ]:


def split_iris_dataset(df):
    '''This function takes in the data from the data frame and splits it into train, validate, test.'''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.species)
    return train, validate, test

def prep_iris_data():
    '''This function reads in the iris dataframe, cleans it and splits it into train, validate, test.'''
    #Define the df
    df= pd.read_csv("iris.csv", index_col=0)
    # Drop the species_id and measurement_id columns
    df = df.drop(columns=['species_id'])
    
    # Rename the species_name column to just species
    df = df.rename(columns={'species_name': 'species'})
    
    # encode the species column
    df_dummies = pd.get_dummies(df[['species']], drop_first=True)
    df = pd.concat([df, df_dummies], axis=1)
    
    # split the data
    train, validate, test = split_iris_dataset(df)
    
    return train, validate, test





def add_scaled_columns(train, validate, test, scaler, columns_to_scale):

    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train_scaled = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index)], axis=1)
    validate_scaled = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index)], axis=1)
    test_scaled = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index)], axis=1)
    
    return train_scaled, validate_scaled, test_scaled

def scale_iris(train, validate, test):
    train_scaled, validate_scaled, test_scaled = add_scaled_columns(
    train,
    validate,
    test,
    scaler=sklearn.preprocessing.MinMaxScaler(),
    columns_to_scale=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    )
    # drop rows not needed for modeling
    cols_to_remove = ['species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    train_scaled = train_scaled.drop(columns=cols_to_remove)
    validate_scaled = validate_scaled.drop(columns=cols_to_remove)
    test_scaled = test_scaled.drop(columns=cols_to_remove)
    return train_scaled, validate_scaled, test_scaled


def wrangle_iris_data():
    """
    This function takes acquired iris data, the cleaned data, scales it
    and splits the data into train, validate, and test datasets
    """
    df = pd.read_csv('iris.csv', index_col=0)
    train, test, validate = prep_iris_data()
    #train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    #train, validate = train_test_split(train_and_validate, test_size=.15, random_state=123)
    # return train, test, validate
    train_scaled, validate_scaled, test_scaled = scale_iris(train, validate, test)
    return train, validate, test, train_scaled, validate_scaled, test_scaled


####### NOTE: to call wrangle_iris_data 
##### train, validate, test, train_scaled, validate_scaled, test_scaled = wrangle_iris_data()

