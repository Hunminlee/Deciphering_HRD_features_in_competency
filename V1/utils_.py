import config
import pyreadstat
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler



def data_import(file_path):
    df, meta = pyreadstat.read_sav(file_path)
    return df, meta

def see_col_idx_and_name(df, meta):
    for i, j in zip(df.columns, meta.column_labels):
        print(i, j)


def normalization(X):
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    columns = X.columns

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    # Reconstruct the DataFrame with the original column names
    X_normalized = pd.DataFrame(X_scaled, columns=columns, index=X.index)

    return X_normalized


def data_aug_smote(X, y):
    X_scaled = normalization(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    print(f"After SMOTE - X: {X_res.shape}, y: {pd.Series(y_res).value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def data_aug_smote_tomek(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X_scaled, y)
    print(f"After SMOTE+Tomek - X: {X_res.shape}, y: {pd.Series(y_res).value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test



def data_undersample(X, y):
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    print(f"After Random Undersampling - X: {X_res.shape}, y: {pd.Series(y_res).value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


#################################################

def Y_remove_Nan(df, target_idx):
    if df[df.columns[target_idx]].isna().sum() > 0:
        df_cleaned = df.dropna(subset=[df.columns[target_idx]])
        return df_cleaned
        
    else:
        return df
    

def HRD_col_select(X, idx):
    config.HRD_idx.append(config.target_col[idx])
    new_X = X[config.HRD_idx]
    return new_X



