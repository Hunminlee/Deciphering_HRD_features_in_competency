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


def analyze_dataframe_step1(df: pd.DataFrame, meta, verbose):
    non_numeric_info = {}
    Meta_col = []
    total_rows = len(df)

    for idx, col in enumerate(df.columns):
        # 숫자 변환 불가능한 값 마스크
        non_numeric_mask = ~pd.to_numeric(df[col], errors='coerce').notna()
        non_numeric_count = non_numeric_mask.sum()
        nan_count = df[col].isna().sum()

        if non_numeric_count > 0:
            non_numeric_info[col] = {
                'non_numeric': non_numeric_count,
                'nan': nan_count
            }
            Meta_col.append(idx)

    # 결과 출력
    if verbose:
        print("\n\n🔍 숫자가 아닌 character가 포함된 컬럼들 및 해당 row 수:")
    for idx, (col, stats) in enumerate(non_numeric_info.items()):
        label = meta.column_labels[Meta_col[idx]]
        if verbose:
            print(f"- {col} ({label}): 숫자 아님 {stats['non_numeric']}개 / NaN {stats['nan']}개")




'''
def clean_dataframe_step1(df: pd.DataFrame, columns_to_drop: list, verbose):
    print("\n=================================\nPreprocessing\n=================================\n")
    total_rows = len(df)

    cols_to_drop = set(columns_to_drop)  # 삭제할 컬럼 집합

    for idx, col in enumerate(df.columns):
        nan_count = df[col].isna().sum()
        nan_ratio = nan_count / total_rows

        # NaN이 25% 미만이면 평균으로 대체
        if nan_count > 0 and nan_ratio < 0.25:
            try:
                mean_val = pd.to_numeric(df[col], errors='coerce').mean()
                df[col] = df[col].fillna(int(mean_val))
                if verbose:
                    print(f"→ {col}: NaN {nan_count}개 평균({int(mean_val):.2f})으로 대체 완료")
            except:
                if verbose:
                    print(f"→ {col}: 평균 계산 불가 (비숫자형 포함 등)")
                pass
        # NaN이 25% 이상이면 삭제 리스트에 추가
        elif nan_ratio >= 0.25:
            if verbose:
                print(f"→ {col}: NaN 비율 {nan_ratio:.2%}로 삭제 대상 추가")
            cols_to_drop.add(col)

    # 컬럼 삭제
    cleaned_df = df.drop(columns=list(cols_to_drop), errors='ignore')

    return cleaned_df'''


