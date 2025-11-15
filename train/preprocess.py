import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def read_dataframe(filename):
    df1_train = pd.read_parquet('../data/final/2022_data.parquet')
    df2_train = pd.read_parquet('../data/final/2023_data.parquet')
    df = pd.read_parquet(filename)
    return df

def preprocess(df, dv, fit_dv=False):
    categorical = ['date', 'dayofweek', 'away_league', 'home_league', 'park_id']
    numerical = ['home_OPS_blend', 'home_FIP_blend', 'home_FPCT_blend', 'away_OPS_blend', 'away_FIP_blend', 'away_FPCT_blend']
    target = ['home_won']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

def run_data_prep(train_data_path, val_data_path, dest_path):
    df_train = read_dataframe(train_data_path)
    df_val = read_dataframe(val_data_path)
    target = ['home_won']
    y_train = df_train[target].values.ravel()
    y_val = df_val[target].values.ravel()

    vec = DictVectorizer(sparse=False)
    X_train, dv = preprocess(df_train, vec, fit_dv=True)
    X_val, _ = preprocess(df_val, dv)
    
    os.makedirs(dest_path, exist_ok=True)
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))

if __name__ == '__main__':
    train_year_list = ['2022', '2023']
    val_year = '2024'
    run_data_prep(train_data_path=[f'../data/final/{year}_data.parquet' for year in train_year_list], \
        val_data_path = f'../data/final/{val_year}_data.parquet', \
        dest_path='../data/vector/')