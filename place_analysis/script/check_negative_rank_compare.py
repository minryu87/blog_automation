import pandas as pd
import glob
import os

def check_negative_rank_compare():
    """
    지정된 디렉토리의 모든 CSV 파일을 검사하여 'place_rank_compare' 컬럼에
    음수 값이 있는 행을 찾아 출력합니다.
    """
    data_dir = 'blog_automation/place_analysis/data/raw_data/지역별_검색순위/'
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

    found_negatives = []

    print(f"'{data_dir}' 경로의 CSV 파일들을 검사합니다...")

    for file_path in sorted(csv_files): # 파일 이름순으로 정렬하여 일관된 결과 출력
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)

            if 'place_rank_compare' not in df.columns:
                continue

            # to_numeric으로 안전하게 숫자 변환
            df['place_rank_compare_numeric'] = pd.to_numeric(df['place_rank_compare'], errors='coerce')

            # 음수 값 필터링
            negative_rows = df[df['place_rank_compare_numeric'] < 0]

            if not negative_rows.empty:
                for index, row in negative_rows.iterrows():
                    found_negatives.append({
                        "file_name": file_name,
                        "place_rank": row.get('place_rank', 'N/A'),
                        "place_name": row.get('place_name', 'N/A'),
                        "place_rank_compare": int(row['place_rank_compare_numeric']) # 정수형으로 출력
                    })
        except Exception as e:
            print(f"'{file_name}' 파일 처리 중 오류 발생: {e}")

    if found_negatives:
        print("\n--- 'place_rank_compare' 컬럼에 음수 값이 포함된 데이터 목록 ---")
        
        result_df = pd.DataFrame(found_negatives)
        result_df.rename(columns={
            'file_name': '파일명',
            'place_rank': '순위 (place_rank)',
            'place_name': '병원명 (place_name)',
            'place_rank_compare': '순위 변화량 (place_rank_compare)'
        }, inplace=True)

        print(result_df.to_markdown(index=False, tablefmt="grid"))

    else:
        print("\n검사한 모든 파일에서 'place_rank_compare' 컬럼에 음수 값을 찾지 못했습니다.")

if __name__ == "__main__":
    check_negative_rank_compare()
