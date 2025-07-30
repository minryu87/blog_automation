import os
import json
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def create_club_info_map(processed_dir):
    """
    processed 폴더의 모든 JSON을 한 번만 스캔하여
    club_id를 key로, cafe_name과 cafe_nickname을 value로 갖는 맵을 생성합니다.
    """
    print("게시글 데이터에서 club_id와 카페 정보를 매핑합니다...")
    club_info_map = {}
    
    if not os.path.exists(processed_dir):
        print(f"오류: '{processed_dir}' 폴더를 찾을 수 없습니다.")
        return club_info_map

    target_files = [f for f in os.listdir(processed_dir) if f.endswith('_analyzed.json')]

    for filename in tqdm(target_files, desc="JSON 파일 스캔 중"):
        file_path = os.path.join(processed_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for post in data:
            club_id = post.get('club_id')
            link = post.get('link')
            
            if club_id and link and club_id not in club_info_map:
                try:
                    # 'https://cafe.naver.com/{cafe_nickname}/...' 형식에서 cafe_nickname 추출
                    cafe_nickname = link.split('/')[3]
                    club_info_map[club_id] = {
                        'cafe_name': post.get('cafe_name'),
                        'cafe_nickname': cafe_nickname
                    }
                except IndexError:
                    # 링크 형식이 예상과 다를 경우 건너뜀
                    continue
                    
    print(f"총 {len(club_info_map)}개의 고유한 카페 정보를 확인했습니다.")
    return club_info_map

def scrape_member_count(cafe_nickname: str) -> int | None:
    """주어진 cafe_nickname으로 카페 페이지를 스크랩하여 회원 수를 반환합니다."""
    url = f"https://cafe.naver.com/{cafe_nickname}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 'mem-cnt-info' 클래스를 가진 li 태그 안의 em 태그를 찾음
        member_element = soup.select_one('li.mem-cnt-info em')
        
        if member_element:
            # "45,039비공개"와 같은 텍스트에서 숫자 부분만 추출
            member_text = member_element.get_text()
            # 정규식을 사용하여 텍스트에서 숫자와 쉼표(,)만 포함된 부분을 추출
            count_match = re.search(r'[\d,]+', member_text)
            if count_match:
                # 쉼표(,)를 제거하고 정수로 변환
                return int(count_match.group().replace(',', ''))

    except requests.RequestException as e:
        print(f"'{cafe_nickname}' 스크래핑 오류: {e}")
    except Exception as e:
        print(f"'{cafe_nickname}' 파싱 오류: {e}")
        
    return None

def main():
    """메인 실행 함수: club_id 목록 확보, 스크래핑, CSV 저장."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cafe_analysis_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(cafe_analysis_dir, 'data', 'historical_processed')
    output_dir = script_dir

    # 1. club_id와 카페 정보 매핑
    club_info_map = create_club_info_map(processed_dir)

    # monthly_view_estimates.csv에서 club_id 목록을 가져와 매핑에 있는 것만 대상으로 함
    try:
        views_df = pd.read_csv(os.path.join(script_dir, 'monthly_view_estimates.csv'))
        # CSV에서 읽은 club_id를 정수형을 거쳐 문자열로 변환하여 데이터 타입을 통일
        unique_club_ids = views_df['club_id'].dropna().astype(int).astype(str).unique()
        print(f"monthly_view_estimates.csv에서 {len(unique_club_ids)}개의 고유 club_id를 찾았습니다.")
    except FileNotFoundError:
        print(f"오류: 'monthly_view_estimates.csv' 파일을 찾을 수 없습니다.")
        return

    final_results = []
    
    # 2 & 3. 각 club_id에 대해 회원수 스크래핑
    for club_id in tqdm(unique_club_ids, desc="카페 회원수 수집 중"):
        if club_id in club_info_map:
            info = club_info_map[club_id]
            member_count = scrape_member_count(info['cafe_nickname'])
            
            if member_count is not None:
                final_results.append({
                    'cafe_name': info['cafe_name'],
                    'cafe_nickname': info['cafe_nickname'],
                    'membercount': member_count
                })
            # 서버에 부담을 주지 않도록 약간의 딜레이
            time.sleep(0.5)

    # 4. CSV 파일로 저장
    if final_results:
        output_df = pd.DataFrame(final_results)
        output_path = os.path.join(output_dir, 'cafe_member_counts.csv')
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n===== 카페 회원수 수집 완료. 결과 저장: {output_path} =====")
    else:
        print("\n수집된 회원수 데이터가 없습니다.")


if __name__ == "__main__":
    main() 