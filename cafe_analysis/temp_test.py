import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

def parse_articles_from_json_html(soup):
    """2페이지 이후 JSON 응답에 포함된 HTML을 위한 전용 파서입니다."""
    articles = []
    # 2페이지 JSON 내 HTML은 'li.bx'가 아닌 'div.view_wrap'을 기준으로 구성될 수 있습니다.
    # 제공해주신 데이터를 기반으로 가장 안정적인 최상위 선택자인 `li.bx._bx`를 사용합니다.
    for li in soup.select('li.bx._bx'):
        title_element = li.select_one('div.title_area > a.title_link')
        link = title_element['href'] if title_element else "링크 없음"
        
        article_id = None
        art_token = None
        if link != "링크 없음":
            try:
                parsed_url = urlparse(link)
                path_parts = parsed_url.path.strip('/').split('/')
                article_id = path_parts[-1] if path_parts and path_parts[-1].isdigit() else None
                
                query_params = parse_qs(parsed_url.query)
                art_token = query_params.get('art', [None])[0]
            except Exception:
                pass

        if not article_id:
            continue

        # 2페이지 HTML 구조에 맞는 댓글 선택자
        comment_elements = li.select('div.comment_box p.txt')
        comment_previews = [c.get_text(strip=True) for c in comment_elements]

        article_data = {
            'title': title_element.get_text(strip=True) if title_element else "제목 없음",
            'link': link,
            'article_id': article_id,
            'art_token': art_token,
            'cafe_name': (li.select_one('div.user_info > a.name').get_text(strip=True) if li.select_one('div.user_info > a.name') else ""),
            'date': (li.select_one('div.user_info > span.sub').get_text(strip=True) if li.select_one('div.user_info > span.sub') else ""),
            'preview': (li.select_one('div.dsc_area > a.dsc_link').get_text(strip=True) if li.select_one('div.dsc_area > a.dsc_link') else ""),
            'comment_preview': comment_previews,
        }
        articles.append(article_data)
    return articles

def main():
    """메인 실행 함수"""
    file_path = "blog_automation/cafe_analysis/data/data_input/step2.txt"
    print(f"--- '{file_path}' 파일 읽기 시작 ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        print("--- 파일 읽기 완료, JSON 파싱 시도 ---")
        json_data = json.loads(raw_content)
        html_part = json_data.get('collection', [{}])[0].get('html', '')

        if not html_part:
            print("오류: JSON 데이터에서 'html' 부분을 찾을 수 없습니다.")
            return

        print("--- HTML 추출 완료, BeautifulSoup으로 파싱 ---")
        soup = BeautifulSoup(html_part, 'lxml')

        print("--- 게시글 파싱 시작 ---")
        articles = parse_articles_from_json_html(soup)

        if articles:
            print(f"\n===== 파싱 성공! 총 {len(articles)}개의 게시글을 찾았습니다. =====")
            print("\n===== 첫 번째 게시글 데이터: =====")
            print(json.dumps(articles[0], indent=4, ensure_ascii=False))
            print("\n===== 마지막 게시글 데이터: =====")
            print(json.dumps(articles[-1], indent=4, ensure_ascii=False))
        else:
            print("\n===== 파싱 실패: 게시글을 하나도 찾지 못했습니다. =====")
            print("HTML 구조나 파싱 로직(CSS 선택자)을 확인해야 합니다.")

    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")
    except (json.JSONDecodeError, IndexError) as e:
        print(f"오류: JSON 데이터를 처리하는 중 문제가 발생했습니다: {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")

if __name__ == '__main__':
    main() 