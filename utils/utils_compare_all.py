import config
import pyreadstat
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def data_import():
    
    df, meta = pyreadstat.read_sav(config.file_path)
    return df, meta



def Y_Nan_check(df, target_idx):
    if df[df.columns[target_idx]].isna().sum() > 0:
        df_cleaned = df.dropna(subset=[df.columns[target_idx]])
        return df_cleaned
        
    else:
        return df
    

def HRD_col_select(X, idx):    
    
    config.HRD_idx.append(config.target_col[idx])
    new_X = X[config.HRD_idx]
    return new_X


    
def data_processing(target_idx):

    df, meta = data_import()
    
    con1 = df[df.columns[target_idx]] == config.middle_val
    con2 = df[df.columns[target_idx]] == config.outlier_val
    cons = con1 | con2
    
    filtered_df = df[~cons]

    filtered_df = Y_Nan_check(filtered_df, target_idx)
    
    nan_values = filtered_df.isna()
    nan_count_per_column = nan_values.sum()
    Nan_cols = []
    columns_with_nan = nan_count_per_column[nan_count_per_column > 0]  # Filter columns with NaN values
    print(f"num of Columns with NaN values and count of NaN values: {len(columns_with_nan)}")
    
    for column, count in columns_with_nan.items():
        #print(f"Column '{column}' has {count} NaN value(7s).")
        Nan_cols.append(column)
    
    no_nan_dataframe = filtered_df.drop(columns=Nan_cols)
    
    no_nan_dataframe[df.columns[target_idx]].value_counts()
    
    # Define the mapping dictionary
    replacement = {1: 0, 2: 0, 4: 1, 5: 1}
    no_nan_dataframe[df.columns[target_idx]] = no_nan_dataframe[df.columns[target_idx]].replace(replacement)
    
    Y = no_nan_dataframe[df.columns[target_idx]]
    X = no_nan_dataframe.drop(columns=df.columns[target_idx])
    #Y.value_counts()

    return X, Y 


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


def data_aug_smote(X, Y):

    X = normalization(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    print(f"Data Augmented - X shape: {X_resampled.shape}, Y shape: {y_resampled.shape}")
    
    print(f"{pd.DataFrame(y_resampled).value_counts()}")
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test