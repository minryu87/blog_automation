import os
import json
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import random
import re
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

import asyncio
import aiohttp
from agno.agent import Agent
from agno.models.google import Gemini
from proxy_manager import get_working_proxies

# --- 로깅 설정 ---
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

# --- LLM 기반 1차 필터링 클래스 (내부로 복귀) ---
class PreFilter:
    """LLM을 사용하여 게시글의 관련성을 1차적으로 필터링하는 클래스"""
    def __init__(self, agent: Agent):
        self.agent = agent
        self.prompt_template = """
        당신은 네이버 카페 게시글의 관련성을 판단하는 전문가입니다.
        주어진 게시글의 '제목', '미리보기', '댓글 미리보기' 내용을 보고, 이 글이 '동탄 지역의 치과'에 대한 후기, 추천, 질문, 정보 공유 등 직접적인 관련이 있는지 판단해주세요.
        치과 이름이 직접 언급되거나, '동탄' 지역임이 명확해야 합니다. 단순 키워드 포함만으로는 안됩니다.
        
        **분석할 데이터:**
        - 제목: {title}
        - 본문 미리보기: {preview}
        - 댓글 미리보기: {comment_preview}

        **응답 형식:**
        - 관련이 있다면: {"is_related": true}
        - 관련이 없다면: {"is_related": false}
        반드시 JSON 형식으로만 응답하고 다른 설명은 추가하지 마세요.
        """
    
    async def is_related(self, title: str, preview: str, comment_preview: str) -> dict:
        if not self.agent:
            raise RuntimeError("LLM Agent is not initialized.")

        if not title and not preview and not comment_preview:
             logging.warning("LLM 필터링 건너뜀: 분석할 내용이 없습니다.")
             return {'answer': json.dumps({"is_related": False})}

        try:
            loop = asyncio.get_event_loop()
            run_with_params = functools.partial(
                self.agent.run,
                name="is_related_to_dongtan_dental_clinic",
                description="Check if the article is related to Dongtan dental clinic reviews or recommendations.",
                prompt=self.prompt_template,
                prompt_params={'title': title, 'preview': preview, 'comment_preview': comment_preview}
            )
            response_dict = await loop.run_in_executor(None, run_with_params)
            return response_dict
        except Exception as e:
            logging.error(f"LLM is_related 호출 실패 (제목: '{title}'): {e}", exc_info=True)
            raise

class HistoricalCafeCrawler:
    """
    특정 기간과 키워드로 네이버 카페 게시글을 크롤링하고,
    페이지 단위로 처리하여 안정성과 재시작 가능성을 확보하는 크롤러
    """
    def __init__(self, start_date, end_date, keyword, test_mode=False):
        self.start_date = start_date
        self.end_date = end_date
        self.keyword = keyword
        
        # .env 파일에서 쿠키 로드
        load_dotenv(find_dotenv())
        self.naver_cookie = os.getenv("NAVER_COOKIE")
        if not self.naver_cookie:
            raise ValueError("환경변수에서 NAVER_COOKIE를 찾을 수 없습니다.")

        # 파일 경로 설정
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_path = os.path.join(self.base_dir, 'data', 'historical_raw', 'checkpoint.json')
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'historical_raw')

        self.club_id_cache = {}
        # LLM 에이전트 초기화
        try:
            llm = Gemini(id=os.getenv("GEMINI_LITE_MODEL", "gemini-1.5-flash-latest"), api_key=os.getenv("GEMINI_API_KEY"))
            agent = Agent(model=llm)
            self.pre_filter_agent = PreFilter(agent) # 내부 PreFilter 인스턴스 생성
        except Exception as e:
            logging.error(f"LLM 에이전트 초기화 실패: {e}")
            self.pre_filter_agent = None

        self.proxies = [] # 초기화
        self.semaphore = asyncio.Semaphore(8) # 동시에 최대 8개 작업만 허용

    async def initialize_proxies(self):
        """ProxyManager를 통해 작동하는 프록시 목록을 비동기로 초기화합니다."""
        self.proxies = await get_working_proxies()
        if not self.proxies:
            logging.warning("작동하는 프록시를 찾지 못했습니다. 프록시 없이 실행을 시도합니다.")

    def _load_checkpoint(self):
        """체크포인트 파일을 로드하여 마지막으로 작업한 위치와 다음 URL을 반환합니다."""
        if not os.path.exists(self.checkpoint_path):
            return None
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                # 필수 키가 모두 있는지 확인
                if all(k in checkpoint_data for k in ['year', 'month', 'page', 'next_page_url']):
                    return checkpoint_data
                return None
        except (json.JSONDecodeError, KeyError):
            logging.warning("체크포인트 파일이 손상되었거나 형식이 잘못되었습니다. 새로 시작합니다.")
            return None

    def _save_checkpoint(self, year, month, page, next_page_url):
        """현재 작업 위치와 다음 페이지 URL을 체크포인트 파일에 저장합니다."""
        checkpoint_data = {
            'year': year,
            'month': month,
            'page': page,
            'next_page_url': next_page_url,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=4)
        logging.info(f"체크포인트 저장됨: {year}-{month}, page {page} 완료. 다음 URL: {str(next_page_url)[:100]}...")

    async def run_async(self):
        """비동기 크롤러 실행"""
        logging.info("===== 네이버 카페 과거 데이터 크롤링 시작 (비동기) =====")
        await self.initialize_proxies()
        if not self.proxies:
            logging.error("작동하는 프록시를 찾지 못했습니다. 프로그램을 종료합니다.")
            return

        checkpoint = self._load_checkpoint()
        
        start_month_dt = self.start_date
        if checkpoint:
            start_month_dt = datetime(checkpoint['year'], checkpoint['month'], 1)

        current_month = start_month_dt

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        if self.naver_cookie:
            headers['Cookie'] = self.naver_cookie

        async with aiohttp.ClientSession(headers=headers) as session:
            while current_month <= self.end_date:
                start_page = 1
                start_url = None
                
                # 현재 작업 월이 체크포인트 월과 같을 때만 페이지/URL 정보 사용
                if checkpoint and checkpoint['year'] == current_month.year and checkpoint['month'] == current_month.month:
                    start_page = checkpoint['page'] + 1
                    start_url = checkpoint['next_page_url']
                    logging.info(f"체크포인트 발견. {current_month.strftime('%Y-%m')}의 {start_page}페이지부터 재시작합니다.")
                    # 사용된 체크포인트는 초기화
                    checkpoint = None
                else:
                    logging.info(f"--- {current_month.year}-{current_month.month} 기간 작업 시작 ---")

                await self._process_month_async(session, current_month.year, current_month.month, start_page, start_url)
                current_month += relativedelta(months=1)
                
                if current_month <= self.end_date:
                    await asyncio.sleep(random.uniform(5, 10))

        logging.info("===== 모든 기간 크롤링 완료 =====")

    async def _process_month_async(self, session, year, month, start_page=1, start_url=None):
        """한 달 단위로 페이지를 순회하며 데이터를 수집하고 처리합니다."""
        page = start_page
        next_page_url = start_url
        response_content = None # response_content 초기화

        if page == 1 and not start_url:
            next_page_url = self._get_start_url(year, month)
            # 1페이지 로딩 재시도 로직
            max_retries = 5
            for attempt in range(max_retries):
                logging.info(f"{year}-{month} 1페이지 로딩 시도 ({attempt + 1}/{max_retries})...")
                response_content = await self._fetch_page_async(session, next_page_url, is_html=True)
                if response_content:
                    break
                logging.warning(f"1페이지 로딩 실패. 잠시 후 다른 프록시로 재시도합니다.")
                await asyncio.sleep(random.uniform(2, 5))
            
            if not response_content:
                logging.error(f"{year}-{month}의 첫 페이지 로딩에 최종 실패했습니다. 이 달의 작업을 건너뜁니다.")
                return
        elif start_url: # 체크포인트에서 재시작하는 경우
             response_content = await self._fetch_page_async(session, next_page_url, is_html=False)


        # 첫 페이지 파싱 (위에서 이미 fetch 했으므로)
        if page == 1:
            if is_html_response := isinstance(response_content, str):
                 articles, next_page_url_extracted = self._parse_articles_from_html(response_content)
            else: # JSON 응답일 경우 (체크포인트 재시작 시)
                 articles, next_page_url_extracted = self._parse_articles_from_json(response_content)
        else: # 2페이지 이상
            articles, next_page_url_extracted = [], None

        if not next_page_url: # 첫 페이지 로딩/파싱 실패 시
            return

        # ... 이하 로직은 거의 동일하나, 루프 시작 전 처리 로직 추가
        if articles:
            processed_articles = await self._process_article_batch(session, articles)
            if processed_articles:
                self._save_processed_data(year, month, page, processed_articles)
            self._save_checkpoint(year, month, page, next_page_url_extracted)
            next_page_url = next_page_url_extracted
            page += 1

        while next_page_url:
            logging.info(f"다음 페이지로 이동합니다. ( {page}페이지 )")
            await asyncio.sleep(random.uniform(3, 7))

            response_content = await self._fetch_page_async(session, next_page_url, is_html=False) # 2페이지부터는 항상 JSON

            if not response_content:
                logging.error(f"{year}-{month} {page}페이지 요청 실패. 이 달의 수집을 중단합니다.")
                break

            articles, next_page_url_extracted = self._parse_articles_from_json(response_content)
            
            if not articles:
                logging.info(f"{year}-{month} {page}페이지에서 게시글을 찾을 수 없습니다. 수집을 종료합니다.")
                self._save_checkpoint(year, month, page, None) # 완료 표시
                break
            
            processed_articles = await self._process_article_batch(session, articles)
            
            if processed_articles:
                self._save_processed_data(year, month, page, processed_articles)
            
            self._save_checkpoint(year, month, page, next_page_url_extracted)
            
            next_page_url = next_page_url_extracted
            page += 1
            
        logging.info(f"{year}-{month} 기간의 모든 페이지 수집 완료.")
                
    def _get_start_url(self, year, month):
        """월의 시작(1페이지) URL을 생성합니다."""
        end_of_month = (datetime(year, month, 1) + relativedelta(months=1)) - timedelta(days=1)
        actual_end_date = min(end_of_month, self.end_date)
        
        date_from_str = datetime(year, month, 1).strftime('%Y%m%d')
        date_to_str = actual_end_date.strftime('%Y%m%d')
        
        return f"https://search.naver.com/search.naver?cafe_where=&date_option=8&prdtype=0&ssc=tab.cafe.all&st=rel&query={self.keyword}&ie=utf8&date_from={date_from_str}&date_to={date_to_str}&srchby=text&dup_remove=1&sm=tab_opt&nso=so%3Ar%2Cp%3Afrom{date_from_str}to{date_to_str}&nso_open=1"

    async def _fetch_page_async(self, session, url, is_html):
        """주어진 URL로 페이지를 요청하고 내용을 반환합니다. 프록시를 사용합니다."""
        proxy = random.choice(self.proxies) if self.proxies else None
        try:
            async with session.get(url, timeout=20, proxy=proxy) as response:
                response.raise_for_status()
                logging.info(f"요청 성공 (프록시: {proxy}, URL: {url[:80]}...)")
                return await response.text() if is_html else await response.json()
        except Exception as e:
            logging.error(f"비동기 페이지 요청 실패 (프록시: {proxy}, URL: {url[:80]}...): {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"JSON 파싱 실패 ({url[:80]}...): {e}")
            return None

    def _parse_article_li(self, li):
        """게시글 li 태그 하나를 파싱하여 dict로 반환합니다."""
        data = {}
        link_tag = li.select_one('a.link_tit')
        if not link_tag: return None
        
        data['title'] = link_tag.get_text(strip=True)
        data['link'] = link_tag['href']
        
        parsed_url = urlparse(data['link'])
        query_params = parse_qs(parsed_url.query)
        data['article_id'] = query_params.get('articleid', [None])[0]
        data['art_token'] = query_params.get('art', [None])[0]

        data['cafe_name'] = li.select_one('a.cafe_name').get_text(strip=True) if li.select_one('a.cafe_name') else ''
        data['date'] = li.select_one('span.sub_time').get_text(strip=True) if li.select_one('span.sub_time') else ''
        data['preview'] = li.select_one('div.dsc_area a.dsc_link').get_text(strip=True) if li.select_one('div.dsc_area a.dsc_link') else ''
        
        comment_elements = li.select('div.sub_area a.cmmt_link .txt')
        data['comment_preview'] = [c.get_text(strip=True) for c in comment_elements]
        
        return data

    def _parse_articles_from_html(self, html_content):
        """HTML 내용에서 게시글 목록과 다음 페이지 URL을 파싱합니다."""
        soup = BeautifulSoup(html_content, 'lxml')
        articles = []
        for li in soup.select('li.bx._bx'):
            article_data = self._parse_article_li(li)
            if article_data:
                articles.append(article_data)

        # 다음 페이지 URL 추출 (중복 문제 해결)
        next_page_url = None
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string and 'new s({url:' in script.string:
                match = re.search(r"""new s\({url:['"](.*?)['"]""", script.string)
                if match:
                    next_api_url = match.group(1).replace('&amp;', '&')
                    if next_api_url.startswith('http'):
                        full_url = next_api_url
                    else:
                        full_url = "https://s.search.naver.com" + next_api_url
                    
                    logging.info(f"다음 페이지 API URL 발견: {full_url}")
                    next_page_url = full_url
                    break
        
        # URL 추출 실패 시 디버깅을 위해 HTML 파일 저장
        if not next_page_url:
            logging.error("다음 페이지 URL을 찾지 못했습니다. HTML 내용을 'debug_page_1.html'에 저장합니다.")
            debug_file_path = os.path.join(self.base_dir, 'debug_page_1.html')
            with open(debug_file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        return articles, next_page_url

    def _parse_articles_from_json(self, json_content):
        """JSON 내용의 HTML에서 게시글 목록을 파싱합니다."""
        if not json_content or 'html' not in json_content:
            return [], None
        
        html_from_json = json_content['html']
        soup = BeautifulSoup(html_from_json, 'lxml')
        articles = []
        for li in soup.select('li'): # JSON의 li는 ul로 감싸여있지 않음
            article_data = self._parse_article_li(li)
            if article_data:
                articles.append(article_data)
        return articles, self._extract_next_page_url_from_json_content(json_content)

    def _extract_next_page_url_from_html_content(self, html_content):
        """HTML 내용에서 다음 페이지의 API URL을 추출합니다."""
        soup = BeautifulSoup(html_content, 'lxml')
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string and 'new s({url:' in script.string:
                match = re.search(r'url:"(https://s\.search\.naver\.com/p/cafe/[^"]+)"', script.string)
                if match:
                    next_api_url = match.group(1).replace('&amp;', '&')
                    logging.info(f"다음 페이지 API URL 발견: {next_api_url}")
                    return next_api_url
        return None

    def _extract_next_page_url_from_json_content(self, json_content):
        """JSON 내용에서 다음 페이지의 API URL을 추출합니다."""
        next_url = json_content.get('url')
        if next_url:
            full_url = "https://s.search.naver.com" + next_url
            logging.info(f"다음 페이지 API URL 발견: {full_url}")
            return full_url
        return None

    async def _process_article_batch(self, session, articles):
        """게시글 배치를 받아 필터링, 상세 정보 수집을 비동기 병렬로 처리합니다."""

        async def _filter_and_enrich(article):
            """LLM으로 필터링하고, 관련 있는 경우에만 상세 정보를 가져옵니다."""
            max_retries = 3
            llm_response_text = "" # 변수 초기화

            for attempt in range(max_retries):
                try:
                    # 1. LLM 호출
                    llm_response_dict = await self.pre_filter_agent.is_related(
                        article.get('title', ''),
                        article.get('preview', ''),
                        json.dumps(article.get('comment_preview', []), ensure_ascii=False)
                    )

                    # 2. 응답에서 JSON 텍스트 추출
                    llm_response_text = llm_response_dict.get('answer', '{}')
                    
                    # 3. JSON 파싱
                    result_json = json.loads(llm_response_text)
                    is_related = result_json.get('is_related', False)
                    article['is_related_analysis'] = result_json
                    
                    # 성공 시 처리
                    if is_related:
                        details = await self._fetch_details_async(session, article)
                        article.update(details)
                        return article
                    else:
                        return None # 관련 없는 게시물

                except json.JSONDecodeError:
                    logging.warning(f"LLM 필터링 JSON 파싱 오류 (시도 {attempt + 1}/{max_retries})")
                    self._log_llm_error(article.get('title', 'N/A'), llm_response_text)
                    if attempt == max_retries - 1:
                        logging.error(f"LLM 필터링 최종 실패: {article['title']}")
                        article['is_related_analysis'] = {"error": "JSONDecodeError", "raw_response": llm_response_text}
                        return None
                    await asyncio.sleep(1)

                except Exception as e:
                    logging.error(f"LLM/상세정보 처리 중 예상치 못한 오류: {e}", exc_info=True)
                    article['is_related_analysis'] = {"error": str(e)}
                    return None
            
            return None
        
        filter_tasks = [_filter_and_enrich(art) for art in articles]
        processed_articles_with_none = await asyncio.gather(*filter_tasks)
        
        processed_articles = [art for art in processed_articles_with_none if art]
        
        logging.info(f"LLM 1차 필터링 및 처리 완료: {len(articles)}개 중 {len(processed_articles)}개 최종 통과.")

        return [art for art in processed_articles if 'club_id' in art]

    async def _fetch_details_async(self, session, article):
        """게시글의 상세 정보(조회수, 댓글 등)를 비동기적으로 가져옵니다."""
        # club_id, art_token 없으면 수집 불가
        if 'club_id' not in article or not article['club_id'] or 'art_token' not in article:
            return article

        max_retries = 3
        # API 호출 전용 프록시 풀
        available_proxies = self.proxies[:] if self.proxies else [None]

        for attempt in range(max_retries):
            proxy_address = random.choice(available_proxies) if available_proxies else None
            proxy = f"http://{proxy_address}" if proxy_address else None
            
            try:
                # v3 API URL 구성
                api_url = f"https://apis.naver.com/cafe-web/cafe-articleapi/v3/cafes/{article['club_id']}/articles/{article['article_id']}?query=&art={article['art_token']}&useCafeId=true&requestFrom=A"
                
                async with session.get(api_url, proxy=proxy, timeout=15) as response:
                    response.raise_for_status()
                    details_json = await response.json()
                    
                    # 조회수, 댓글 등 상세 정보 파싱 및 추가
                    article['read_count'] = details_json.get('result', {}).get('readCount', 0)
                    
                    comments = details_json.get('result', {}).get('comments', {}).get('items', [])
                    new_comments = []
                    for c in comments:
                        # 타임스탬프를 날짜 문자열로 변환
                        timestamp_ms = c.get('updateDate', 0)
                        if isinstance(timestamp_ms, int):
                            dt_object = datetime.fromtimestamp(timestamp_ms / 1000)
                            formatted_date = dt_object.strftime('%Y.%m.%d.')
                            new_comments.append({'date': formatted_date, 'content': c.get('content','').strip()})
                    article['comments'] = new_comments
                    return article # 성공

            except Exception as e:
                logging.warning(f"상세 정보 수집 실패 (시도 {attempt + 1}/{max_retries}, 프록시: {proxy}): {article['title'][:20]}... - {e}")
                
                if proxy_address in available_proxies:
                    available_proxies.remove(proxy_address)
                    logging.info(f"API 접근 실패 프록시 {proxy} 제거. 남은 API 프록시: {len(available_proxies)}개")

                if not available_proxies:
                    logging.error("API 수집용 프록시 소진. 상세 정보 수집 중단.")
                    break
                
                await asyncio.sleep(random.uniform(1, 3))

        logging.error(f"최종 상세 정보 수집 실패: {article['title'][:20]}...")
        return article

    def _save_processed_data(self, year, month, page, articles):
        """처리 완료된 한 페이지의 데이터를 파일로 저장합니다."""
        if not articles: return
        
        filename = f"{year:04d}-{month:02d}_page_{page:03d}_processed.json"
        filepath = os.path.join(self.raw_data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
        logging.info(f"{len(articles)}개의 상세 데이터를 '{filepath}'에 저장했습니다.")

    def _log_llm_error(self, article_title, original_response):
        """LLM 오류를 별도의 로그 파일에 기록합니다."""
        log_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, 'llm_errors.log')
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"===== LLM Error: {timestamp} =====\n"
        log_entry += f"Article Title: {article_title}\n"
        log_entry += f"LLM Response:\n---\n{original_response}\n---\n\n"
        
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)


if __name__ == '__main__':
    async def main():
        crawler = HistoricalCafeCrawler(
            start_date=datetime(2024, 6, 1),
            end_date=datetime(2025, 7, 1),
            keyword="동탄 치과"
        )
        await crawler.run_async()

    asyncio.run(main()) 