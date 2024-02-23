import matplotlib.pyplot as plt
from numpy import nanpercentile
import pandas as pd

def remove_outliers_iqr(data, columns: list):
    ''' Calculate outliers  based on IQR and returns df  without outliers'''
    
    for column in columns:
        print(f"Column name: {column}")
        # calculate interquartile range
        q25, q75 = nanpercentile(data[column], 25), nanpercentile(data[column], 75)
        iqr = q75 - q25

        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        print("lower cutoff", lower)
        print("upper cutoff", upper)

        # Remove outliers from df by indexing
        outliers_mask = (data[column] < lower) | (data[column] > upper)
        outliers = data[outliers_mask]

        # Remove outliers from the DataFrame
        data_no_outliers = data[~outliers_mask]
        
    return data_no_outliers

# def mod_z_score(df, value:str):
#     abs_value_median = abs(df[value] - df[value].median())
#     mad = abs_value_median.median()
#     df["modified_z_score"] = 0.6745 * ((df[value] - df[value].median()) / mad)
    
#     # Here we select values that are 3.5 higher than median value, 3.5 is a general rule of thumb
#     z_mask = (df.modified_z_score > 3.5) | (df.modified_z_score < -3.5)
#     data_no_outliers = df[~z_mask]
    
#     return data_no_outliers

def mod_z_score(df, columns:list):
    df_no_outliers = df.copy()

    for column in columns:
        print(f"Feature: {column}")
        abs_column_median = abs(df_no_outliers[column] - df_no_outliers[column].median())
        mad = abs_column_median.median()
        print(f"MAD: {mad}")
        df_no_outliers[column+"_modified_z_score"] = 0.6745 * ((df_no_outliers[column] - df_no_outliers[column].median()) / mad)
        
        # Here we select values that are 3.5 higher than median value, 3.5 is a general rule of thumb
        z_mask = (df_no_outliers[column+"_modified_z_score"] > 3.5) | (df_no_outliers[column+"_modified_z_score"] < -3.5)
        df_no_outliers = df_no_outliers[~z_mask]  # Update df_no_outliers with outliers removed
    
    return df_no_outliers

