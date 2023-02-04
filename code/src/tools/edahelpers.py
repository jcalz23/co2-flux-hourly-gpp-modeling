import pandas as pd

def get_min_max(df):
    return (df.min(), df.max())

def get_min_max_datetime(df):
    return (pd.to_datetime(df).min(), pd.to_datetime(df).max())

def is_leap_year(year):
    return year%4 == 0 ;