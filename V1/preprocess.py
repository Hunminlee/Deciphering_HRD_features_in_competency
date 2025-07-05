import os
import numpy as np
import pandas as pd
import config
import utils_


def get_all_data(year_list, file_names):
    # Expected file names (e.g., ['file1.csv', 'file2.csv'])
    file_list = os.listdir(config.path)  # All files in the directory
    dataset = {}                         # Final dictionary to store data
    year_cnt = -1                        # Counter to map years to files

    for expected_file in file_names:
        for actual_file in file_list:
            if expected_file == actual_file:
                year_cnt += 1
                current_year = year_list[year_cnt]
                print(f"{expected_file} ===> {current_year} data")

                # Load data
                df, meta = utils_.data_import(os.path.join(config.path, expected_file))

                # Store in dataset dict with year as key
                dataset[current_year] = {
                    'data': df,
                    'meta': meta
                }

    return dataset


###############################


def filter_dataset_by_common_ids(dataset, years = config.year_list):
    id_sets = []

    for year in years:
        col_name = f"W{year[2:]}ID1"
        df = dataset[year]['data']  # dataset이 Series라도 동일하게 접근 가능
        ids = set(df[col_name].dropna().unique())
        id_sets.append(ids)

    common_ids = set.intersection(*id_sets)
    print(f"공통 기업 ID 개수: {len(common_ids)}")

    for year in years:
        col_name = f"W{year[2:]}ID1"
        df = dataset[year]['data']
        filtered_df = df[df[col_name].isin(common_ids)].copy()
        dataset[year]['data'] = filtered_df
        print(f"{year} 필터링 후 행 개수: {filtered_df.shape[0]}")

    return dataset


def unify_columns_by_base_name(dataset, years = config.year_list):
    col_sets = []
    base_col_maps = {}

    print("원본 컬럼 수 (with prefix):")
    for year in years:
        df = dataset[year]['data']
        prefix = f"W{year[2:]}"
        base_cols = {col[len(prefix):]: col for col in df.columns if col.startswith(prefix)}
        base_col_maps[year] = base_cols
        col_sets.append(set(base_cols.keys()))

        print(f"  - {year}: {len(df.columns)}개 컬럼 (prefix '{prefix}')")

    # 공통 base 이름 컬럼 추출
    common_base_cols = set.intersection(*col_sets)
    print(f"\n공통 base 컬럼 수: {len(common_base_cols)}개")

    # 각 연도 데이터에서 공통 컬럼만 유지
    print("\n정리된 각 연도별 데이터프레임 shape:")
    for year in years:
        df = dataset[year]['data']
        prefix = f"W{year[2:]}"
        keep_cols = [base_col_maps[year][col] for col in common_base_cols]
        dataset[year]['data'] = df[keep_cols].copy()
        print(f"  - {year}: {df.shape[1]} → {len(keep_cols)}개 컬럼 유지, shape: {dataset[year]['data'].shape}")

    return dataset



def clean_dataframe_step1(df: pd.DataFrame, columns_to_drop: list, verbose=True):
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
    return cleaned_df


def clean_all_years(data_dict, columns_to_drop=[], verbose=True):
    cleaned_dict = {}

    for year in sorted(data_dict.keys()):
        print(f"\n\n=== {year}년 데이터 전처리 ===")
        df = data_dict[year]['data']
        cleaned_df = clean_dataframe_step1(df.copy(), columns_to_drop, verbose=verbose)

        # 결과 저장
        cleaned_dict[year] = {
            'data': cleaned_df,
            'meta': data_dict[year].get('meta', None)
        }

        print(f"{year}년 전처리 완료: 원본 컬럼 수 {df.shape[1]} → 정제 후 컬럼 수 {cleaned_df.shape[1]}\n")

    return cleaned_dict


def merge_worker_head_labels(dataset_Work, dataset_Head, year_w, year_h):
    # Define ID column names
    label_col=f'C{year_h[2:]}C05_01H2'
    work_id_col = f'W{year_w[2:]}ID1'
    head_id_col = f'C{year_h[2:]}_ID1'
    worker_id_col = f'W{year_w[2:]}ID2'
    work_company_id_col = f'W{year_w[2:]}ID3'

    # Extract data
    df_work = dataset_Work[year_w]['data']
    df_head = dataset_Head[year_h]['data']

    # Select only ID and label column from head
    df_head_selected = df_head[[head_id_col, label_col]].copy()

    # Merge on company ID
    merged = pd.merge(df_work, df_head_selected, left_on=work_id_col, right_on=head_id_col, how='inner')

    # Drop ID columns from features
    merged = merged.drop(columns=[work_id_col, head_id_col, worker_id_col, work_company_id_col], errors='ignore')
    #y = merged[label_col]

    print(f"Year: {year_w} | Merged shape: {merged.shape} | work shape: {df_work.shape} | head shape: {df_head_selected.shape}")

    return merged, label_col #, X, y


def clean_target_classes(df: pd.DataFrame, target_col) -> pd.DataFrame:

    df = df[df[target_col] != 3].copy()  # 3인 행 삭제
    df = df[df[target_col] != 8].copy()  # 8인 행 삭제
    df = df[df[target_col] != -8].copy()  # -8인 행 삭제
    df.loc[df[target_col].isin([1, 2]), target_col] = 0 # 1,2 -> 0
    df.loc[df[target_col].isin([4, 5]), target_col] = 1 # 4,5 -> 1
    y = df[target_col]

    df = df.drop(columns=[target_col], errors='ignore')

    #df = standardize_feature_names(df)

    return df.reset_index(drop=True), np.array(y)


import re

def standardize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove year-specific prefixes like W20, W21, W22, W23, etc. from column names.
    """
    def remove_year_prefix(col):
        return re.sub(r'^W\d{2}', '', col)  # Removes 'W20', 'W21', etc.

    df = df.copy()
    df.columns = [remove_year_prefix(col) for col in df.columns]
    return df
