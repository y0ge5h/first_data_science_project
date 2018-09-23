# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 10:31:44 2018

@author: Yogesh
kNN implementations""" 

import pandas as pd
import numpy as np

def convert_price_column_to_float(dataframe):
    dataframe['price'] = dataframe['price'].str.replace(",","").str.replace("$","")
    dataframe['price'] = dataframe['price'].astype(float)
    return dataframe    

def get_numreic_columns(dataframe):
    numeric_df = dataframe._get_numeric_data()
    return numeric_df
    
def replace_missing_with_mean(dataframe):
    numeric_cols = dataframe.columns[dataframe.dtypes==float]
    for x in numeric_cols:
        dataframe[x] = dataframe[x].fillna(dataframe[x].mean())
    return dataframe

def calculate_distance(dataframe , feature1 , feature1_value , feature2 , feature2_value):
    numeric_cols = dataframe.columns[(dataframe.dtypes==float) | (dataframe.dtypes==int)]
    if feature1 in numeric_cols and feature2  in numeric_cols:
        dataframe['distance'] = ((dataframe[feature1] - feature1_value)**2+(dataframe[feature2] - feature2_value)**2)**0.5
    return dataframe
    
def get_k_nearest_neighbours(k , dataframe , feature1 , feature1_value , feature2 , feature2_value):
    
    distance_dataframe = calculate_distance(dataframe , feature1 , feature1_value , feature2 , feature2_value)
    
    rand_distance_dataframe = distance_dataframe.loc[np.random.permutation(len(distance_dataframe))]
    
    dataframe_with_distance_zero = rand_distance_dataframe[(rand_distance_dataframe['distance']==0)]
    
    k_neighbours_df = dataframe_with_distance_zero[:k]
    
    #print(k_neighbours_df['distance'])
    return dataframe_with_distance_zero , k_neighbours_df

def main():
    k = 20
    airbnb = pd.read_csv("D:/grey work/data_sets/dc_airbnb.csv")
    new_airbnb = convert_price_column_to_float(airbnb) 
    filled_airbnb = replace_missing_with_mean(new_airbnb)
   
    distance_zero_df , neighbours = get_k_nearest_neighbours(k,filled_airbnb , "bedrooms" , 2 , "bathrooms" , 2)
    
    neighbours_mean = neighbours['price'].mean()

    distance_zero_df['error squared'] = np.square(distance_zero_df['price'] - neighbours_mean)
    
    RMSE = np.sqrt(distance_zero_df['error squared'].mean())
    
    print(RMSE)
    #print(neighbours['distance'])