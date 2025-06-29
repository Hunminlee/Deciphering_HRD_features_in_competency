import config
import pyreadstat
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

def data_import():
    #path = 'C:/Users/hml76/Desktop/Jupyter/HRD project2/HCCPⅡ_free/'
    #file_path = path+'2. SPSS/HCCP_2ndWave_Head_2nd(최종).sav'
    
    df, meta = pyreadstat.read_sav(config.file_path)
    return df, meta


def Y_Nan_check(df, target_name):
    if df[target_name].isna().sum() > 0:
        df_cleaned = df.dropna(subset=[target_name])
        return df_cleaned
        
    else:
        return df

def HRD_col_select(X, idx):    
    
    #HRD_idx = ['C21C01_01', 'C21C01_02', 'C21C01_03', 'C21C01_04A1','C21C01_04B1','C21C01_04C1','C21C01_04A2','C21C01_04B2','C21C01_04C2','C21C01_05','C21C01_06A1',
    #'C21C01_06A2','C21C01_06B1','C21C01_06B2','C21C01_07','C21C01_07A1','C21C01_07A2','C21C01_07B1','C21C01_07B2','C21C01_07C1','C21C01_07C2',
    #'C21C01_07D1','C21C01_07D2','C21C01_07E1','C21C01_07E2','C21C01_07F1','C21C01_07F2','C21C01_07G1','C21C01_07G2','C21C01_08A','C21C01_08B',
    #'C21C01_08C','C21C02_01A','C21C02_01B','C21C02_01B1','C21C02_01B2','C21C02_01C','C21C02_01D','C21C02_01D1','C21C02_01D2','C21C02_02A1','C21C02_02A2',
    #'C21C02_02B1','C21C02_02B2','C21C02_02C1','C21C02_02C2']
    
    #target_col = ['C21C05_01B2', 'C21C05_01C2', 'C21C05_01D2', 'C21C05_01F2', 'C21C05_01G2', 'C21C05_01H2']

    config.HRD_idx.append(config.target_col[idx])
    new_X = X[config.HRD_idx]
    return new_X


def data_processing(target_name, idx):

    df, meta = data_import()

    df = HRD_col_select(df, idx)
    
    con1 = df[target_name] == config.middle_val
    con2 = df[target_name] == config.outlier_val
    cons = con1 | con2
    
    filtered_df = df[~cons]

    filtered_df = Y_Nan_check(filtered_df, target_name)
    
    nan_values = filtered_df.isna()
    nan_count_per_column = nan_values.sum()
    Nan_cols = []
    columns_with_nan = nan_count_per_column[nan_count_per_column > 0]  # Filter columns with NaN values
    print(f"num of Columns with NaN values and count of NaN values: {len(columns_with_nan)}")
    
    for column, count in columns_with_nan.items():
        #print(f"Column '{column}' has {count} NaN value(7s).")
        Nan_cols.append(column)
    
    no_nan_dataframe = filtered_df.drop(columns=Nan_cols)
    
    no_nan_dataframe[target_name].value_counts()
    
    # Define the mapping dictionary
    replacement = {1: 0, 2: 0, 4: 1, 5: 1}
    no_nan_dataframe[target_name] = no_nan_dataframe[target_name].replace(replacement)
    
    Y = no_nan_dataframe[target_name]
    X = no_nan_dataframe.drop(columns=target_name)
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
    return X_resampled, y_resampled



def count_val():
    pd.Series(config.FIs).value_counts()