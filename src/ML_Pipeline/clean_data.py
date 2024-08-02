import pandas as pd
from ML_Pipeline.constants import *

# Removed unused columns
def remove_unused_columns(df, columns):
    return df.drop(columns, axis=1)

# Replace null values with ""
def null_processing(df):
    # Drop rows with null value in the 'fraudulent' column
    df = df.dropna(axis=0, subset = label)

    # Replace null value with ""
    df.fillna("", inplace=True)
    return df

def clean_data(df, columns):
    df = remove_unused_columns(df, columns)
    df = null_processing(df)
    return df