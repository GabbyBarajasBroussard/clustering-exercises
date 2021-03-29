#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import acquire


# In[9]:


zillow= acquire.get_zillow_data()
zillow=zillow.drop(columns='Unnamed: 0')


# In[10]:


zillow.head()


# In[11]:


zillow.shape


# In[12]:


zillow.info()


# #### Write a function that takes in a dataframe and returns a dataframe with 3 columns: the number of columns missing, percent of columns missing, and number of rows with n columns missing. Run the function and document takeaways from this on how you want to handle missing values.

# In[13]:


def table_information(df):
    
    '''This function will read in a dataframe and returns a new dataframe with the nuber of columns missing, percent of columns missing, and number of rows with n columns missing. '''
    num_rows = df.loc[:].isnull().sum()
    num_cols_missing = df.loc[:, df.isna().any()].count()
    pct_cols_missing = round(df.loc[:, df.isna().any()].count() / len(df.index) * 100, 3)
    missing_cols_and_rows_df = pd.DataFrame({'number_of_columns_missing': num_cols_missing,
                                             'percent_of_columns_missing': pct_cols_missing,
                                             'number_of_rows': num_rows})
    missing_cols_and_rows_df = missing_cols_and_rows_df.fillna(0)
    missing_cols_and_rows_df['number_of_columns_missing'] = missing_cols_and_rows_df['number_of_columns_missing'].astype(int)
    return missing_cols_and_rows_df


# In[14]:


table_information(zillow)


# #### Remove any properties that are likely to be something other than single unit properties. (e.g. no duplexes, no land/lot, ...). There are multiple ways to estimate that a property is a single unit, and there is not a single "right" answer. But for this exercise, do not purely filter by unitcnt as we did previously. Add some new logic that will reduce the number of properties that are falsely removed. You might want to use # bedrooms, square feet, unit type or the like to then identify those with unitcnt not defined.

# In[19]:


unit_df = zillow[zillow['unitcnt']==1]
room_df = zillow[zillow['roomcnt']>0]
garage_df = zillow[zillow['garagecarcnt']>0]
bed_df = zillow[zillow['bedroomcnt']>0]
bath_df = zillow[zillow['bathroomcnt']>0]
p261_df = zillow[zillow['propertylandusetypeid'] == 261]
p263_df = zillow[zillow['propertylandusetypeid'] == 263]
p264_df = zillow[zillow['propertylandusetypeid'] == 264]    
p266_df = zillow[zillow['propertylandusetypeid'] == 266] 
p273_df = zillow[zillow['propertylandusetypeid'] == 273]
p275_df = zillow[zillow['propertylandusetypeid'] == 275]
p276_df = zillow[zillow['propertylandusetypeid'] == 276]    
p279_df = zillow[zillow['propertylandusetypeid'] == 279]
single_unit_df = pd.concat([unit_df, room_df, garage_df,bed_df,bath_df,p261_df, p263_df, p264_df, p266_df, p273_df, p275_df, p276_df, p279_df]).drop_duplicates('id').reset_index(drop=True)
single_unit_df.shape


# In[20]:


def remove_non_single_unit_props (zillow):
    unit_df = zillow[zillow['unitcnt']==1]
    room_df = zillow[zillow['roomcnt']>0]
    garage_df = zillow[zillow['garagecarcnt']>0]
    bed_df = zillow[zillow['bedroomcnt']>0]
    bath_df = zillow[zillow['bathroomcnt']>0]
    p261_df = zillow[zillow['propertylandusetypeid'] == 261]
    p263_df = zillow[zillow['propertylandusetypeid'] == 263]
    p264_df = zillow[zillow['propertylandusetypeid'] == 264]    
    p266_df = zillow[zillow['propertylandusetypeid'] == 266] 
    p273_df = zillow[zillow['propertylandusetypeid'] == 273]
    p275_df = zillow[zillow['propertylandusetypeid'] == 275]
    p276_df = zillow[zillow['propertylandusetypeid'] == 276]    
    p279_df = zillow[zillow['propertylandusetypeid'] == 279]
    single_unit_df = pd.concat([unit_df, room_df, garage_df,bed_df,bath_df,p261_df, p263_df, p264_df, p266_df, p273_df, p275_df, p276_df, p279_df]).drop_duplicates('id').reset_index(drop=True)
    return single_unit_df


# In[24]:


single_unit_df= remove_non_single_unit_props(zillow)


# Create a function that will drop rows or columns based on the percent of values that are missing: handle_missing_values(df, prop_required_column, prop_required_row).
# 
# The input:
# <ol>
# <li>A dataframe
# <li>A number between 0 and 1 that represents the proportion, for each column, of rows with non-missing values required to keep the column. i.e. if prop_required_column = .6, then you are requiring a column to have at least 60% of values not-NA (no more than 40% missing).
# <li>A number between 0 and 1 that represents the proportion, for each row, of columns/variables with non-missing values required to keep the row. For example, if prop_required_row = .75, then you are requiring a row to have at least 75% of variables with a non-missing value (no more that 25% missing).
#     </ol>
# 
# The output:
# <ol>
# <li>The dataframe with the columns and rows dropped as indicated. Be sure to drop the columns prior to the rows in your function.
#     </ol>
# hint:
# <ul>
# <li> Look up the dropna documentation.
# <li>You will want to compute a threshold from your input values (prop_required) and total number of rows or columns.
# <li>Make use of inplace, i.e. inplace=True/False.
# <li>Decide how to handle the remaining missing values:
# </ul>
# Fill with constant value.
# Impute with mean, median, mode.
# Drop row/column

# In[ ]:


def prep_zillow (df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    def remove_columns(df, cols_to_remove):  
        df = df.drop(columns=cols_to_remove)
        return df
    def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
        threshold = int(round(prop_required_column*len(df.index),0))
        df.dropna(axis=1, thresh=threshold, inplace=True)
        threshold = int(round(prop_required_row*len(df.columns),0))
        df.dropna(axis=0, thresh=threshold, inplace=True)
        return df
    df = remove_columns(df, cols_to_remove)  # Removes Specified Columns
    df = handle_missing_values(df, prop_required_column, prop_required_row) # Removes Specified Rows
    #df.dropna(inplace=True) # Drops all Null Values From Dataframe
    return df


# In[25]:


prep_zillow(single_unit_df,
    cols_to_remove=[],
    prop_required_column=.6,
    prop_required_row=.75 )    


# In[ ]:




