import os
import json
import re
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# --- 1. 날짜 정제 함수 ---
def refine_date(date_str: str, base_date: datetime) -> str:
    """다양한 형식의 날짜 문자열을 'YYYY.MM.DD' 형식으로 통일합니다."""
    if not isinstance(date_str, str):
        return base_date.strftime('%Y.%m.%d')

    try:
        # 'YYYY-MM-DD' 형식
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str.replace('-', '.')
        # 'YYYY.MM.DD.' 형식
        match = re.match(r'(\d{4})\.(\d{2})\.(\d{2})\.', date_str)
        if match:
            return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
        # 'N시간 전' 형식
        match = re.match(r'(\d+)시간 전', date_str)
        if match:
            return base_date.strftime('%Y.%m.%d')
        # 'N일 전' 형식
        match = re.match(r'(\d+)일 전', date_str)
        if match:
            return (base_date - timedelta(days=int(match.group(1)))).strftime('%Y.%m.%d')
        # 'N주 전' 형식
        match = re.match(r'(\d+)주 전', date_str)
        if match:
            return (base_date - timedelta(weeks=int(match.group(1)))).strftime('%Y.%m.%d')
    except Exception:
        # 예외 발생 시 안전하게 오늘 날짜 반환
        return base_date.strftime('%Y.%m.%d')

    # 위 형식에 모두 해당하지 않으면 원본 또는 오늘 날짜 반환
    return date_str if '.' in date_str else base_date.strftime('%Y.%m.%d')

# --- 2. 병원명 표준화 함수 ---
def refine_clinic_names(post: dict, mapping: dict):
    """
    mentioned_clinics와 clinic_sentiments 내의 clinic_name을
    매핑 기준으로 표준화하고, 매핑에 없으면 삭제합니다.
    """
    if 'analysis' not in post:
        return

    # 1. mentioned_clinics 표준화
    if 'mentioned_clinics' in post['analysis']:
        original_clinics = post['analysis'].get('mentioned_clinics')
        if isinstance(original_clinics, list):
            refined_clinics = set() # 중복 제거를 위해 set 사용
            for clinic in original_clinics:
                if clinic in mapping:
                    refined_clinics.add(mapping[clinic])
            post['analysis']['mentioned_clinics'] = sorted(list(refined_clinics))

    # 2. clinic_sentiments 내 clinic_name 표준화
    if 'clinic_sentiments' in post['analysis']:
        original_sentiments = post['analysis'].get('clinic_sentiments')
        if isinstance(original_sentiments, list):
            refined_sentiments = []
            for sentiment in original_sentiments:
                # sentiment가 dict이고 'clinic_name' 키를 가졌는지 확인
                if isinstance(sentiment, dict) and 'clinic_name' in sentiment:
                    raw_name = sentiment.get('clinic_name')
                    if raw_name in mapping:
                        # 매핑에 있으면 대표명으로 변경하고 리스트에 추가
                        sentiment['clinic_name'] = mapping[raw_name]
                        refined_sentiments.append(sentiment)
            # 매핑에 있는 병원만 남은 리스트로 교체
            post['analysis']['clinic_sentiments'] = refined_sentiments


# --- 3. 진료 분야 표준화 함수 ---
def refine_treatments(post: dict, mapping: dict):
    """related_treatments를 매핑 기준으로 표준화합니다."""
    if 'analysis' not in post or 'related_treatments' not in post['analysis']:
        return
        
    original_treatments = post['analysis']['related_treatments']
    if not isinstance(original_treatments, list):
        return

    refined_treatments = set()
    for treatment in original_treatments:
        # 매핑에 있으면 대표값으로, 없으면 원본값 유지
        refined_treatments.add(mapping.get(treatment, treatment))
            
    post['analysis']['related_treatments'] = sorted(list(refined_treatments))

def main():
    """데이터 정제 파이프라인 메인 실행 함수."""
    print("===== 데이터 정제 및 표준화 파이프라인 시작 =====")

    # 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cafe_analysis_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(cafe_analysis_dir, 'data', 'historical_processed')
    
    # 매핑 파일 로드
    try:
        clinic_map_path = os.path.join(script_dir, 'transform_clinicname.csv')
        treatment_map_path = os.path.join(script_dir, 'transform_treatment.csv')
        
        clinic_df = pd.read_csv(clinic_map_path)
        treatment_df = pd.read_csv(treatment_map_path)

        # DataFrame을 딕셔너리로 변환하여 검색 성능 향상
        clinic_mapping = clinic_df.set_index('raw_clinicname')['representing_clinicname'].to_dict()
        treatment_mapping = treatment_df.set_index('raw_treantment')['representing_treatment'].to_dict()
        print("병원명 및 진료 분야 매핑 파일 로드 완료.")
    except FileNotFoundError as e:
        print(f"[오류] 매핑 파일을 찾을 수 없습니다: {e}")
        return
    except Exception as e:
        print(f"[오류] 매핑 파일 로드 중 오류 발생: {e}")
        return

    # 처리 대상 파일 목록
    if not os.path.exists(processed_dir):
        print(f"[오류] 처리할 데이터 폴더를 찾을 수 없습니다: {processed_dir}")
        return
        
    target_files = [f for f in os.listdir(processed_dir) if f.endswith('_analyzed.json')]
    if not target_files:
        print("정제할 파일이 없습니다.")
        return

    # 날짜 계산 기준
    base_date = datetime.now()

    # 메인 루프
    print(f"총 {len(target_files)}개의 파일을 처리합니다.")
    for filename in tqdm(target_files, desc="데이터 정제 중"):
        file_path = os.path.join(processed_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for post in data:
                # 1. 날짜 정제
                original_date = post.get('date', '')
                post['date'] = refine_date(original_date, base_date)

                # 2. 병원명 표준화
                refine_clinic_names(post, clinic_mapping)

                # 3. 진료 분야 표준화
                refine_treatments(post, treatment_mapping)

            # 수정된 내용으로 파일 덮어쓰기
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"\n[경고] '{filename}' 파일 처리 중 오류 발생: {e}")
            continue
            
    print("\n===== 모든 데이터 정제 및 표준화 작업이 완료되었습니다. =====")

if __name__ == "__main__":
    main() 