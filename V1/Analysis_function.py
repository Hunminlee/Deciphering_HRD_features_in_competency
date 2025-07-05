
import config



def target_variable_check(dataset, target_variable):
    for year in config.year_list:
        df, meta = dataset[year]['data'], dataset[year]['meta']
        #col_names = [f'C{year[2:]}C05_01H1', f'C{year[2:]}C05_01H2']
        #col_names = [f'C{year[2:]}C05_01H2']
        col_names = [f'W{year[2:]}{target_variable}']
        for col_name in col_names:
            nan_count = df[col_name].isna().sum()
            total = df.shape[0]
            print(f"{year} - {col_name} : NaN {nan_count}개 / 전체 {total}개 ({nan_count / total:.2%})")
            print(df[f'{col_name}'].value_counts())




def compare_company_ids_in_dataset(dataset):
    years = list(dataset.keys())
    id_sets = {}

    for year in years:
        col_name = f"W{year[2:]}ID1"
        df = dataset[year]['data']
        id_sets[year] = set(df[col_name].dropna().unique())

    for i in range(len(years)):
        for j in range(i+1, len(years)):
            y1, y2 = years[i], years[j]
            common = id_sets[y1].intersection(id_sets[y2])
            print(f"{y1} & {y2} 교집합 기업 수: {len(common)}")

    return id_sets


def get_common_company_ids_all_years(dataset):
    years = config.year_list
    id_sets = []

    for year in years:
        col_name = f"W{year[2:]}ID1"
        df = dataset[year]['data']
        ids = set(df[col_name].dropna().unique())
        id_sets.append(ids)

    common_ids = set.intersection(*id_sets)
    print(f"2020~2023 모든 연도에 공통으로 존재하는 기업 ID 수: {len(common_ids)}")

    return common_ids



def compare_company_ids(dataset_Work, dataset_Head, year):
    work_ids = dataset_Work[year]['data'][f'W{year[2:]}ID1']
    head_ids = dataset_Head[year]['data'][f'C{year[2:]}_ID1']

    print(f"🔍 {year} 데이터 비교")

    # 1. 타입 체크
    print(f"- Work ID type: {work_ids.dtype}")
    print(f"- Head ID type: {head_ids.dtype}")

    # 2. 예시 출력
    #print(f"- 예시 Work ID: {work_ids.iloc[:5].tolist()}")
    #print(f"- 예시 Head ID: {head_ids.iloc[:5].tolist()}")

    # 3. 고유 개수
    print(f"- 고유 Work 회사 수: {work_ids.nunique()}")
    print(f"- 고유 Head 회사 수: {head_ids.nunique()}")

    # 4. 교집합 확인
    overlap = set(work_ids.unique()) & set(head_ids.unique())
    print(f"- 공통 회사 ID 수: {len(overlap)}")
    print(f"- 전체 Head ID 중 비율: {len(overlap)/len(head_ids.unique()):.2%}")

    return overlap