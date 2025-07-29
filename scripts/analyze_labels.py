import pandas as pd
import os
from collections import Counter

def analyze_labels():
    """
    라벨링된 검색어 데이터를 분석하여 라벨 분포를 출력합니다.
    """
    # 경로 설정
    workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(workspace_path, 'data', 'data_processed', 'labeled_search_queries.csv')

    # 데이터 로드
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"오류: 분석할 파일 '{input_file}'을 찾을 수 없습니다.")
        print("먼저 label_search_queries.py를 실행하여 라벨링된 파일을 생성해주세요.")
        return
    except Exception as e:
        print(f"오류: 파일을 읽는 중 문제가 발생했습니다: {e}")
        return

    # 'labels' 컬럼이 없는 경우 처리
    if 'labels' not in df.columns:
        print("오류: 'labels' 컬럼이 파일에 존재하지 않습니다.")
        return

    print("라벨 분포 분석을 시작합니다...")

    # NaN 값을 빈 문자열로 대체 후 라벨 분리
    all_labels = df['labels'].dropna().str.split(';').sum()
    
    # 라벨 빈도수 계산
    label_counts = Counter(all_labels)
    
    # 빈 문자열 라벨 제거
    if '' in label_counts:
        del label_counts['']

    # 결과를 데이터프레임으로 변환하여 출력
    count_df = pd.DataFrame(label_counts.most_common(), columns=['Label', 'Count'])
    count_df['Category'] = count_df['Label'].apply(lambda x: x.split(':')[0])
    count_df['Label'] = count_df['Label'].apply(lambda x: x.split(':')[1])
    
    # 카테고리별로 정렬하고, 카운트 순으로 정렬
    count_df = count_df.sort_values(by=['Category', 'Count'], ascending=[True, False]).reset_index(drop=True)

    # 터미널에 결과 출력
    print("""
--- 전체 라벨 분포 요약 ---""")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 100):
        print(count_df)

    # CSV 파일로 저장
    output_file = os.path.join(workspace_path, 'data', 'data_processed', 'label_distribution.csv')
    # Category, Label 순으로 컬럼 순서 재정렬 후 저장
    count_df_to_save = count_df[['Category', 'Label', 'Count']]
    count_df_to_save.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"""
분석 결과가 다음 파일에 저장되었습니다:
{output_file}""")


if __name__ == "__main__":
    analyze_labels() 