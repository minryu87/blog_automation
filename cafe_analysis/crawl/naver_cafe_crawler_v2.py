import os
import json
import re
import time
import random
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import urllib.parse
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

# --- Helper Functions ---

def get_html_with_cookie(url, cookie):
    """지정된 URL에 쿠키를 포함하여 GET 요청을 보내고 HTML을 반환합니다."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Cookie': cookie
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"URL 요청 중 오류 발생 ({url}): {e}")
        return None

def get_json_with_cookie(url, cookie):
    """지정된 URL에 쿠키를 포함하여 GET 요청을 보내고 JSON을 반환합니다. (Referer/User-Agent 없음)"""
    headers = {
        'Cookie': cookie
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"JSON 요청 중 오류 발생 ({url}): {e}")
        return None
    except json.JSONDecodeError:
        print(f"JSON 파싱 오류 ({url}). 응답 내용: {response.text}")
        return None

def extract_article_id_from_url(url):
    """URL에서 articleId를 추출합니다."""
    match = re.search(r'/(\d+)', url.split('?')[0])
    return match.group(1) if match else None

# --- Main Logic Steps ---

def step1_initial_crawl(queries_df, cookie, output_file):
    """1단계: 검색어 기반으로 초기 게시글 정보를 파싱하여 JSON 파일로 저장합니다."""
    print("--- 1단계: 기본 게시글 정보 수집 시작 ---")
    all_results = {}
    for query in tqdm(queries_df['query'], desc="1단계: 검색어 처리 중"):
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://search.naver.com/search.naver?ssc=tab.cafe.all&query={encoded_query}&sm=tab_opt&nso=so:dd,p:3m"
        
        html_content = get_html_with_cookie(search_url, cookie)
        if not html_content:
            all_results[query] = []
            continue

        soup = BeautifulSoup(html_content, 'lxml')
        articles_data = []
        for li in soup.select('li.bx._bx'):
            title_element = li.select_one('a.title_link')
            link = title_element['href'] if title_element else "링크 없음"
            article_id = extract_article_id_from_url(link)
            if not article_id:
                continue

            # 'art' 토큰 값을 URL에서 추출합니다.
            parsed_url = urlparse(link)
            query_params = parse_qs(parsed_url.query)
            art_token = query_params.get('art', [None])[0]

            # 댓글 미리보기 영역의 모든 댓글을 리스트로 수집합니다.
            comment_elements = li.select('div.txt_area')
            comment_previews = [c.text.strip().replace('\n', ' ').replace('\t', ' ') for c in comment_elements]

            articles_data.append({
                'query': query,
                'cafe_name': (li.select_one('a.name').text.strip() if li.select_one('a.name') else ""),
                'post_date': (li.select_one('span.sub').text.strip() if li.select_one('span.sub') else ""),
                'title': (title_element.text.strip() if title_element else ""),
                'preview': (li.select_one('a.dsc_link').text.strip() if li.select_one('a.dsc_link') else ""),
                'comment_preview': comment_previews,
                'link': link,
                'article_id': article_id,
                'art_token': art_token
            })
        all_results[query] = articles_data
        print(f"\n'{query}' 검색 결과: {len(articles_data)}개 게시글 발견.")
        time.sleep(random.uniform(0.7, 1.5))
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"--- 1단계 완료: '{output_file}' 파일 저장 ---")
    return all_results

def step2_get_club_ids(crawled_data, cookie, output_file):
    """2단계: 각 게시글 링크에 접속하여 club_id를 추출하고 JSON 파일로 저장합니다. (카페별 1회 조회 최적화)"""
    print("\n--- 2단계: Club ID 추출 시작 (최적화) ---")
    
    all_articles = [article for articles in crawled_data.values() for article in articles]
    cafe_club_ids = {} # 카페 URL별로 club_id를 저장하는 딕셔너리

    for article in tqdm(all_articles, desc="2단계: Club ID 추출 중"):
        article_url = article.get('link')
        if not article_url:
            article['club_id'] = None
            continue
        
        # 카페의 base URL 추출 (e.g., https://cafe.naver.com/dongtantwomom)
        cafe_base_url_match = re.match(r"(https://cafe\.naver\.com/[^/]+)", article_url)
        if not cafe_base_url_match:
            article['club_id'] = None
            continue
        cafe_base_url = cafe_base_url_match.group(1)

        # 이미 해당 카페의 club_id를 찾았다면, 저장된 값을 사용
        if cafe_base_url in cafe_club_ids:
            article['club_id'] = cafe_club_ids[cafe_base_url]
            continue
        
        # 처음 보는 카페라면, 네트워크 요청으로 club_id 추출
        time.sleep(random.uniform(1.0, 2.0))
        article_html = get_html_with_cookie(article_url, cookie)

        if not article_html:
            club_id = None
        else:
            club_id_match = re.search(r'g_sClubId\s*=\s*"(\d+)"', article_html)
            club_id = club_id_match.group(1) if club_id_match else None
        
        # 찾은 club_id를 현재 게시글과 딕셔너리에 모두 저장
        article['club_id'] = club_id
        cafe_club_ids[cafe_base_url] = club_id


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(crawled_data, f, ensure_ascii=False, indent=4)
    print(f"--- 2단계 완료: '{output_file}' 파일 저장 ---")
    return crawled_data

def step3_enrich_details(data_with_club_ids, cookie, output_file):
    """3단계: 두 가지 다른 JSON 구조에 모두 대응하여 상세 정보를 가져옵니다."""
    print("\n--- 3단계: 상세 정보 (조회수, 댓글) 수집 시작 ---")
    
    all_articles = [article for articles in data_with_club_ids.values() for article in articles]
    
    for article in tqdm(all_articles, desc="3단계: 상세 정보 수집 중"):
        club_id = article.get('club_id')
        article_id = article.get('article_id')
        art_token = article.get('art_token')

        if not club_id or not article_id or not art_token:
            article['views'] = '추출 실패 (ID 또는 토큰 없음)'
            article['comments'] = []
            continue
        
        # v3 API와 art 토큰을 사용하는 새로운 URL 구조
        api_url = f"https://apis.naver.com/cafe-web/cafe-articleapi/v3/cafes/{club_id}/articles/{article_id}?query=&art={art_token}&useCafeId=true&requestFrom=A"
        headers = {'Cookie': cookie}
        
        try:
            time.sleep(random.uniform(0.7, 1.5))
            response = requests.get(api_url, headers=headers, timeout=15)
            response.raise_for_status()
            details = response.json()

            if 'result' not in details:
                raise ValueError("API 응답에 'result' 키가 없습니다.")
            
            result_data = details['result']
            
            # 두 종류의 JSON 구조에 대응하기 위한 분기 처리
            if 'article' in result_data:
                # [구조 1] 'article' 키가 있는 경우 (정상 응답)
                article_data = result_data.get('article', {})
                article['views'] = article_data.get('readCount', 0)
                comments_data = result_data.get('comments', {}).get('items', [])
            else:
                # [구조 2] 'article' 키가 없는 경우 (제한된 정보 응답)
                article['views'] = result_data.get('readCount', 0)
                comments_data = [] # 이 구조에서는 댓글 목록이 없음

            new_comments = []
            for comment in comments_data:
                timestamp = comment.get('updateDate', 0) / 1000
                new_comments.append({
                    'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    'content': comment.get('content', '').strip()
                })
            article['comments'] = new_comments

        except requests.exceptions.RequestException as e:
            print(f"\n[API 요청 실패] {article.get('title', '')}: {e}")
            article['views'] = 'API 요청 실패'
            article['comments'] = []
        except (json.JSONDecodeError, ValueError) as e:
            print(f"\n[JSON 처리 실패] {article.get('title', '')}: {e}")
            article['views'] = 'JSON 처리 실패'
            article['comments'] = []
        except Exception as e:
            print(f"\n[알 수 없는 오류] {article.get('title', '')}: {e}")
            article['views'] = '알 수 없는 오류'
            article['comments'] = []

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_with_club_ids, f, ensure_ascii=False, indent=4)
    print(f"--- 3단계 완료: 최종 결과 '{output_file}' 파일 저장 ---")
    return data_with_club_ids


def main():
    """메인 실행 함수"""
    # .env 파일에서 쿠키 값을 로드합니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    naver_cookie = os.getenv('NAVER_COOKIE')
    
    # 디버깅을 위해 사용 중인 쿠키 값을 출력합니다.
    print("-" * 50)
    print(f"사용 중인 쿠키: {naver_cookie}")
    print("-" * 50)

    if not naver_cookie or naver_cookie == "여기에_네이버_쿠키_값을_붙여넣으세요":
        print(f"오류: '{dotenv_path}' 파일에 유효한 NAVER_COOKIE를 설정해주세요.")
        return
    
    # 입/출력 파일 경로 설정
    queries_file = os.path.join(script_dir, 'search_queries.csv')
    step1_output_file = os.path.join(script_dir, 'step1_initial_crawl_v2.json')
    step2_output_file = os.path.join(script_dir, 'step2_with_club_id_v2.json')
    final_output_file = os.path.join(script_dir, 'naver_cafe_crawl_result_v2.json')

    try:
        queries_df = pd.read_csv(queries_file)
    except FileNotFoundError:
        print(f"오류: '{queries_file}' 파일을 찾을 수 없습니다.")
        return

    # 각 단계를 순차적으로 실행
    # 이전 단계의 결과 파일이 있으면 로드해서 다음 단계를 실행합니다.
    if not os.path.exists(step1_output_file):
        step1_data = step1_initial_crawl(queries_df, naver_cookie, step1_output_file)
    else:
        print(f"--- 1단계 건너뛰기: '{step1_output_file}' 파일이 이미 존재합니다. ---")
        with open(step1_output_file, 'r', encoding='utf-8') as f:
            step1_data = json.load(f)

    if not os.path.exists(step2_output_file):
        step2_data = step2_get_club_ids(step1_data, naver_cookie, step2_output_file)
    else:
        print(f"--- 2단계 건너뛰기: '{step2_output_file}' 파일이 이미 존재합니다. ---")
        with open(step2_output_file, 'r', encoding='utf-8') as f:
            step2_data = json.load(f)

    step3_enrich_details(step2_data, naver_cookie, final_output_file)
    
    print("\n\n🎉 모든 크롤링 과정이 완료되었습니다! 🎉")
    print(f"최종 결과는 '{final_output_file}' 파일에서 확인하실 수 있습니다.")

if __name__ == '__main__':
    main() 
    main() 
