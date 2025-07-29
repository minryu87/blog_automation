import pandas as pd
import numpy as np

def create_search_console_analysis():
    """
    네이버 검색량 데이터와 블로그 유입량 데이터를 병합하여
    검색어별 월별 검색 점유율을 분석하는 통합 데이터 파일을 생성합니다.
    """
    # 파일 경로 정의
    volume_file = 'blog_automation/data/data_processed/keyword_search_volume_transform.csv'
    inflow_file = 'blog_automation/searchquery_analysis/data/search_inflow_by_month.csv'
    output_file = 'blog_automation/searchquery_analysis/data/search_console_analysis.csv'

    # 데이터 불러오기
    print("데이터 파일을 불러오는 중입니다...")
    df_volume = pd.read_csv(volume_file)
    df_inflow = pd.read_csv(inflow_file)
    print("데이터 불러오기 완료.")

    # --- 데이터 전처리 ---
    print("데이터 전처리를 시작합니다...")

    # 1. df_volume 전처리
    # 'searchQuery' 컬럼명을 'searchquery'로 변경하여 통일
    df_volume.rename(columns={'searchQuery': 'searchquery'}, inplace=True)
    # PC, Mobile 검색량 합산하여 'total_search_volume' 생성
    df_volume['total_search_volume'] = df_volume['pc'] + df_volume['mobile']
    # 필요한 컬럼만 선택
    df_volume_processed = df_volume[['searchquery', 'date', 'total_search_volume']]

    # 2. df_inflow 전처리
    # 'date' 컬럼을 datetime 형식으로 변환 후 'YYYY-MM' 형식으로 통일
    df_inflow['date'] = pd.to_datetime(df_inflow['date']).dt.strftime('%Y-%m')

    print("데이터 전처리 완료.")

    # --- 데이터 병합 ---
    print("데이터 병합을 시작합니다...")
    # inflow 데이터를 기준으로 volume 데이터를 left-join
    merged_df = pd.merge(df_inflow, df_volume_processed, on=['searchquery', 'date'], how='left')
    print("데이터 병합 완료.")

    # --- 규칙 적용 및 계산 ---
    print("규칙 적용 및 계산을 시작합니다...")
    # '임의검색량설정여부' 컬럼 생성 (기본값 0)
    merged_df['arbitrary_volume_flag'] = 0
    
    # total_search_volume이 없는 경우 (NaN) 마스크 생성
    missing_volume_mask = merged_df['total_search_volume'].isnull()

    # 규칙 적용: 검색량 없으면 5로 설정, 플래그도 5로 설정
    merged_df.loc[missing_volume_mask, 'total_search_volume'] = 5
    merged_df.loc[missing_volume_mask, 'arbitrary_volume_flag'] = 5

    # 검색 점유율 계산 (0으로 나누는 경우 방지)
    merged_df['search_share'] = np.where(
        merged_df['total_search_volume'] > 0,
        merged_df['searchinflow'] / merged_df['total_search_volume'],
        0
    )
    print("규칙 적용 및 계산 완료.")

    # --- 최종 데이터프레임 정리 ---
    print("최종 데이터프레임을 정리합니다...")
    # 컬럼명 변경
    final_df = merged_df.rename(columns={
        'searchquery': '검색어',
        'date': '월',
        'total_search_volume': '네이버 전체 검색량',
        'searchinflow': '우리 블로그 검색 유입량',
        'search_share': '검색 점유율',
        'arbitrary_volume_flag': '임의검색량설정여부'
    })

    # 최종 컬럼 순서 지정
    final_df = final_df[[
        '검색어',
        '월',
        '네이버 전체 검색량',
        '우리 블로그 검색 유입량',
        '검색 점유율',
        '임의검색량설정여부'
    ]]
    print("데이터프레임 정리 완료.")

    # --- 파일 저장 ---
    print(f"'{output_file}' 파일로 저장 중입니다...")
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("파일 저장 완료!")

    return output_file


def create_aggregated_analysis(input_file):
    """
    월별 분석 데이터를 검색어 기준으로 통합하고,
    검색 점유율을 재계산하여 요약 리포트를 생성합니다.
    """
    output_file = 'blog_automation/searchquery_analysis/data/search_console_analysis_aggregated.csv'
    
    print(f"'{input_file}' 파일을 불러와 통합 분석을 시작합니다...")
    df = pd.read_csv(input_file)
    print("파일 불러오기 완료.")

    # --- 데이터 통합 (Aggregation) ---
    print("데이터 통합을 시작합니다...")
    agg_df = df.groupby('검색어').agg(
        네이버_전체_검색량_합=('네이버 전체 검색량', 'sum'),
        우리_블로그_검색_유입량_합=('우리 블로그 검색 유입량', 'sum'),
        임의검색량설정여부=('임의검색량설정여부', 'max') # 기간 중 한 번이라도 임의 설정되었으면 5로 표시
    ).reset_index()
    print("데이터 통합 완료.")

    # --- 검색 점유율 재계산 ---
    print("검색 점유율을 재계산합니다...")
    agg_df['검색 점유율'] = np.where(
        agg_df['네이버_전체_검색량_합'] > 0,
        agg_df['우리_블로그_검색_유입량_합'] / agg_df['네이버_전체_검색량_합'],
        0
    )
    print("검색 점유율 재계산 완료.")
    
    # --- 컬럼명 변경 및 정렬 ---
    print("최종 데이터프레임을 정리합니다...")
    agg_df.rename(columns={
        '네이버_전체_검색량_합': '네이버 전체 검색량 합',
        '우리_블로그_검색_유입량_합': '우리 블로그 검색 유입량 합'
    }, inplace=True)
    
    # 최종 컬럼 순서 지정
    final_agg_df = agg_df[[
        '검색어',
        '네이버 전체 검색량 합',
        '우리 블로그 검색 유입량 합',
        '검색 점유율',
        '임의검색량설정여부'
    ]]
    
    # 검색 점유율 기준 내림차순 정렬
    final_agg_df = final_agg_df.sort_values(by='검색 점유율', ascending=False)
    print("데이터프레임 정리 및 정렬 완료.")

    # --- 파일 저장 ---
    print(f"'{output_file}' 파일로 저장 중입니다...")
    final_agg_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("통합 분석 파일 저장 완료!")


if __name__ == '__main__':
    monthly_analysis_file = create_search_console_analysis()
    create_aggregated_analysis(monthly_analysis_file) 