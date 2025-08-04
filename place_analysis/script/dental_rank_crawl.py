import requests
import pandas as pd
import json
import os
from tqdm import tqdm
import time

# --- Configuration ---
API_URL = 'https://adlog.kr/adlog/naver_place_list_exec.php'
SEARCH_QUERY_PATH = 'blog_automation/place_analysis/data/raw_data/지역별_검색순위/searchquery.txt'
# V2: 저장 경로 분리
RAW_OUTPUT_DIR = 'blog_automation/place_analysis/data/raw_data/지역별_검색순위'
PROCESSED_OUTPUT_DIR = 'blog_automation/place_analysis/data/processed_data/지역별_검색순위'

# --- API Information (Hardcoded) ---
HEADERS = {
    'Cookie': 'adsession=5ld3h8gndps9egievm9jiapcto; adsession=5ld3h8gndps9egievm9jiapcto; _fbp=fb.1.1754274398576.835500299409487831; _ga=GA1.1.555874785.1754274399; _fwb=48tsjDqdDYzOTUNiijvyaH.1754274492408; 2a0d2363701f23f8a75028924a3af643=MTE0LjIwNC41OC43Nw%3D%3D; 2cefeb2fd91f206b1b78b5b081f490af=MTE0LjIwNC41OC43Nw%3D%3D; ck_font_resize_rmv_class=; ck_font_resize_add_class=; wcs_bt=22be812e885a68:1754321439; _ga_PQ5HJPKPY1=GS2.1.s1754320066$o6$g1$t1754321658$j55$l0$h0'
}

BODY_TEMPLATE = {
    'api_section': '10',
    'api_type': '95',
    'n': 'h',
    'display': '50'
}

# --- Helper Functions ---
def clean_dataframe(df):
    """API 응답으로 생성된 데이터프레임을 정제합니다."""
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].notna().any():
            df[col] = df[col].astype(str).str.replace(r'<[^>]+>', '', regex=True).str.replace(',', '', regex=False).str.strip()

    if 'place_rank_compare' in df.columns:
        def parse_rank_compare(value):
            """순위 변화값을 올바른 정수형으로 변환합니다 ('1▼' -> -1, '10▲' -> 10)."""
            val_str = str(value).strip()
            if '▼' in val_str:
                return f"-{val_str.replace('▼', '')}"
            cleaned_val = val_str.replace('▲', '').replace('-', '0').strip()
            return cleaned_val if cleaned_val else '0'
        df['place_rank_compare'] = df['place_rank_compare'].apply(parse_rank_compare)

    for col in df.columns:
        if df[col].dtype == 'object':
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if not numeric_col.isna().all():
                 df[col] = numeric_col.fillna(0)
    
    return df

# --- Main Logic ---
def main():
    """메인 크롤링 및 저장 로직"""
    # V2: 경로 생성
    os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_OUTPUT_DIR, exist_ok=True)
    
    try:
        with open(SEARCH_QUERY_PATH, 'r', encoding='utf-8') as f:
            search_queries = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"오류: 검색어 파일을 읽는 중 문제가 발생했습니다 - {e}")
        return

    # 이어하기 기능: 처리된 CSV 파일 목록 확인
    already_crawled = [f.replace('.csv', '') for f in os.listdir(PROCESSED_OUTPUT_DIR) if f.endswith('.csv')]
    queries_to_crawl = [q for q in search_queries if q not in already_crawled]
    
    print(f"총 {len(search_queries)}개 검색어 중 {len(already_crawled)}개는 이미 수집되었습니다.")
    print(f"{len(queries_to_crawl)}개의 신규 검색어에 대한 수집을 시작합니다.")

    # 크롤링 루프
    for query in tqdm(queries_to_crawl, desc="지역별 순위 데이터 수집 중"):
        try:
            data = BODY_TEMPLATE.copy()
            data['keyword'] = query
            
            response = requests.post(API_URL, headers=HEADERS, data=data, timeout=30)
            response.raise_for_status()

            response_data = response.json()
            
            # V2: 원본 JSON 응답 저장
            raw_output_path = os.path.join(RAW_OUTPUT_DIR, f"{query}.json")
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=4)
            
            if response_data.get('code') != '0000' or not response_data.get('items'):
                error_msg = response_data.get('msg', '결과 없음')
                print(f"\n'{query}'에 대한 API 오류: {error_msg}")
                if '일일 사용 수량' in error_msg:
                    print("\n일일 API 사용량을 초과하여 수집을 중단합니다.")
                    break
                continue

            df = pd.DataFrame(response_data['items'])
            if not df.empty:
                df = clean_dataframe(df)

                # V2: 처리된 CSV 파일 저장
                processed_output_path = os.path.join(PROCESSED_OUTPUT_DIR, f"{query}.csv")
                df.to_csv(processed_output_path, index=False, encoding='utf-8-sig')
            
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"\n'{query}'에 대한 네트워크 오류 발생: {e}")
        except json.JSONDecodeError:
            print(f"\n'{query}'에 대한 JSON 파싱 오류: 서버 응답이 올바른 JSON이 아닙니다.")
            print(f"응답 내용 일부: {response.text[:200]}...")
        except Exception as e:
            print(f"\n'{query}' 처리 중 예기치 않은 오류 발생: {e}")
    
    print("\n모든 크롤링 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
