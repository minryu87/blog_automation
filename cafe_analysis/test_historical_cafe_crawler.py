import os
import json
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import random
import re
from urllib.parse import urlparse, parse_qs
import asyncio
import aiohttp
from agno.agent import Agent
from agno.models.google import Gemini

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from proxy_manager import get_working_proxies
import functools

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
        
        # LLM 호출 전 내용이 비어있는지 확인
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
    def __init__(self, start_date, end_date, keyword, test_mode=True):
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
        # 테스트 모드일 경우 체크포인트 파일명 변경
        checkpoint_filename = 'checkpoint_test.json' if test_mode else 'checkpoint.json'
        self.checkpoint_path = os.path.join(self.base_dir, 'data', 'historical_raw', checkpoint_filename)
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'historical_raw')
        self.test_mode = test_mode

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
        """한 달 단위의 크롤링을 조율하고, 1페이지와 2페이지 이후 과정을 분리하여 처리합니다."""
        # 체크포인트가 없거나 1페이지부터 시작해야 하는 경우
        if start_page == 1 and not start_url:
            logging.info(f"[{year}-{month}] 1페이지(HTML) 수집을 시작합니다.")
            next_page_url_from_page1 = await self._process_first_page_async(session, year, month)
            
            if not next_page_url_from_page1:
                logging.warning(f"{year}-{month}의 1페이지 처리에서 다음 페이지 URL을 얻지 못했습니다. 월 단위 작업을 종료합니다.")
                return

            # 2페이지부터 이어서 처리
            start_page = 2
            start_url = next_page_url_from_page1
        else:
            # 체크포인트가 있는 경우, 해당 페이지부터 바로 시작
            logging.info(f"체크포인트에 따라 [{year}-{month}]의 {start_page}페이지(JSON)부터 시작합니다.")

        # 2페이지 이후 처리 (start_url이 유효할 때만)
        if start_url:
            await self._process_subsequent_pages_async(session, year, month, start_page, start_url)
        else:
            logging.info(f"[{year}-{month}] 기간은 1페이지만 존재하거나, 유효한 다음 페이지가 없어 처리를 종료합니다.")
        
        logging.info(f"--- {year}-{month} 기간 작업 완료 ---")

    async def _process_first_page_async(self, session, year, month):
        """1페이지(HTML)를 수집 및 처리하고, 2페이지의 API URL을 반환합니다."""
        page = 1
        start_url = self._get_start_url(year, month)

        response_text = await self._fetch_page_async(session, start_url)
        if not response_text:
            logging.error(f"[{year}-{month} {page}페이지] HTML 로딩에 실패했습니다.")
            return None

        articles, next_page_url = self._parse_articles_from_html(response_text)
        
        await self._process_article_batch(session, articles, year, month, page)
        
        self._save_checkpoint(year, month, page, next_page_url)
        
        if not next_page_url:
            logging.info(f"[{year}-{month}] 기간은 1페이지만 존재합니다.")

        return next_page_url

    async def _process_subsequent_pages_async(self, session, year, month, start_page, start_url):
        """2페이지 이후(JSON)의 모든 페이지를 순회하며 수집하고 처리합니다."""
        page = start_page
        current_url = start_url

        while current_url:
            response_text = await self._fetch_page_async(session, current_url)
            if not response_text:
                logging.error(f"[{year}-{month} {page}페이지] JSON 로딩에 실패했습니다. 이 달의 작업을 중단합니다.")
                break

            articles, next_page_url = self._parse_articles_from_json(response_text)

            await self._process_article_batch(session, articles, year, month, page)
            
            self._save_checkpoint(year, month, page, next_page_url)
            
            if not next_page_url:
                logging.info(f"[{year}-{month}] 기간의 모든 페이지 수집을 완료했습니다.")
                break

            current_url = next_page_url
            page += 1
            logging.info(f"다음 페이지로 이동합니다. ( {page}페이지 )")
            await asyncio.sleep(random.uniform(3, 7))

    def _get_start_url(self, year, month):
        """월의 시작(1페이지) URL을 생성합니다."""
        end_of_month = (datetime(year, month, 1) + relativedelta(months=1)) - timedelta(days=1)
        actual_end_date = min(end_of_month, self.end_date)
        
        date_from_str = datetime(year, month, 1).strftime('%Y%m%d')
        date_to_str = actual_end_date.strftime('%Y%m%d')
        
        return f"https://search.naver.com/search.naver?cafe_where=&date_option=8&prdtype=0&ssc=tab.cafe.all&st=rel&query={self.keyword}&ie=utf8&date_from={date_from_str}&date_to={date_to_str}&srchby=text&dup_remove=1&sm=tab_opt&nso=so%3Ar%2Cp%3Afrom{date_from_str}to{date_to_str}&nso_open=1"

    async def _fetch_page_async(self, session, url):
        """항상 응답을 순수 텍스트(str)로만 반환하도록 역할을 단순화합니다."""
        if not url:
            logging.warning("요청할 URL이 없습니다.")
            return None
        proxy = f"http://{random.choice(self.proxies)}" if self.proxies else None
        try:
            async with session.get(url, proxy=proxy, timeout=20) as response:
                response.raise_for_status()
                logging.info(f"요청 성공 (프록시: {proxy}, URL: {str(url)[:120]}...)")
                # 어떠한 가정도 하지 않고, 순수 텍스트만 반환합니다.
                return await response.text()
        except Exception as e:
            logging.error(f"비동기 페이지 요청 실패 (프록시: {proxy}, URL: {str(url)[:120]}...): {e}")
            if proxy in self.proxies:
                self.proxies.remove(proxy)
                logging.warning(f"작동하지 않는 프록시 {proxy}를 목록에서 제거합니다. 남은 프록시: {len(self.proxies)}개")
            return None

    def _parse_article_li(self, soup):
        """BeautifulSoup 객체에서 li 태그들을 파싱하여 게시글 데이터 리스트를 반환합니다."""
        articles = []
        for li in soup.select('li.bx._bx'):
            title_element = li.select_one('a.title_link')
            link = title_element['href'] if title_element else "링크 없음"
            
            article_id = None
            art_token = None
            if link != "링크 없음":
                try:
                    parsed_url = urlparse(link)
                    # article_id는 URL 경로의 마지막 숫자 부분으로 가정
                    article_id_match = re.search(r'/(\d+)$', parsed_url.path)
                    if article_id_match:
                        article_id = article_id_match.group(1)
                    
                    query_params = parse_qs(parsed_url.query)
                    art_token = query_params.get('art', [None])[0]
                except Exception:
                    # 링크 구조가 예상과 다를 경우를 대비
                    pass

            # article_id가 없으면 유효한 게시물로 보지 않음
            if not article_id:
                continue

            comment_elements = li.select('div.txt_area')
            comment_previews = [c.text.strip().replace('\n', ' ').replace('\t', ' ') for c in comment_elements]

            article_data = {
                'title': title_element.text.strip() if title_element else "제목 없음",
                'link': link,
                'article_id': article_id,
                'art_token': art_token,
                'cafe_name': (li.select_one('a.name').text.strip() if li.select_one('a.name') else ""),
                'date': (li.select_one('span.sub').text.strip() if li.select_one('span.sub') else ""),
                'preview': (li.select_one('a.dsc_link').text.strip() if li.select_one('a.dsc_link') else ""),
                'comment_preview': comment_previews,
            }
            articles.append(article_data)
        return articles

    def _parse_article_li_from_json_html(self, soup):
        """2페이지 이후 JSON 응답에 포함된 HTML을 위한 전용 파서입니다."""
        articles = []
        for li in soup.select('li.bx._bx'):
            title_element = li.select_one('a.title_link')
            link = title_element['href'] if title_element else "링크 없음"
            
            article_id = None
            art_token = None
            if link != "링크 없음":
                try:
                    parsed_url = urlparse(link)
                    article_id_match = re.search(r'/(\d+)', parsed_url.path)
                    if article_id_match:
                        article_id = article_id_match.group(1)
                    
                    query_params = parse_qs(parsed_url.query)
                    art_token = query_params.get('art', [None])[0]
                except Exception:
                    pass

            if not article_id:
                continue

            comment_elements = li.select('div.comment_area > div.comment_box p.txt')
            comment_previews = [c.text.strip().replace('\n', ' ').replace('\t', ' ') for c in comment_elements]
            
            article_data = {
                'title': title_element.text.strip() if title_element else "제목 없음",
                'link': link,
                'article_id': article_id,
                'art_token': art_token,
                'cafe_name': (li.select_one('div.user_info > a.name').text.strip() if li.select_one('div.user_info > a.name') else ""),
                'date': (li.select_one('div.user_info > span.sub').text.strip() if li.select_one('div.user_info > span.sub') else ""),
                'preview': (li.select_one('div.dsc_area > a.dsc_link').text.strip() if li.select_one('div.dsc_area > a.dsc_link') else ""),
                'comment_preview': comment_previews,
            }
            articles.append(article_data)
        return articles

    def _parse_articles_from_html(self, html_content):
        """HTML 내용에서 게시글 목록과 다음 페이지 URL을 파싱합니다."""
        soup = BeautifulSoup(html_content, 'lxml')
        articles = self._parse_article_li(soup)

        # 게시글 파싱 실패 시 디버깅용 HTML 저장
        if not articles:
            logging.error("1페이지에서 게시글을 하나도 찾지 못했습니다. 디버깅을 위해 'debug_page_1_articles_not_found.html'에 HTML을 저장합니다.")
            with open("debug_page_1_articles_not_found.html", "w", encoding="utf-8") as f:
                f.write(html_content)

        # 다음 페이지 URL 추출 (중복 문제 해결)
        next_page_url = None
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string and 'new s({url:' in script.string:
                match = re.search(r"""new s\({url:['"](.*?)['"]""", script.string)
                if match:
                    next_api_url = match.group(1).replace('&amp;', '&')
                    
                    parsed = urlparse(next_api_url)
                    if parsed.scheme and parsed.netloc:
                        full_url = next_api_url
                    elif next_api_url.startswith('/'):
                        full_url = "https://s.search.naver.com" + next_api_url
                    else:
                        full_url = "https://s.search.naver.com/" + next_api_url.lstrip('/')
                    
                    # 최종 중복 방지
                    if full_url.count('https://s.search.naver.com') > 1:
                        full_url = "https://s.search.naver.com" + full_url.split("https://s.search.naver.com")[-1]

                    logging.info(f"다음 페이지 API URL 발견: {full_url}")
                    next_page_url = full_url
                    break
        
        if not next_page_url:
            logging.error("다음 페이지 URL을 찾지 못했습니다. HTML 내용을 'debug_page_1.html'에 저장합니다.")
            with open("debug_page_1.html", "w", encoding="utf-8") as f:
                f.write(html_content)
                
        return articles, next_page_url

    def _parse_articles_from_json(self, response_text):
        """JSON 텍스트를 받아 파싱합니다. 이제 이 함수가 JSON 변환을 책임집니다."""
        try:
            # 이 함수의 첫 단계는 항상 문자열을 JSON 딕셔너리로 변환하는 것입니다.
            json_data = json.loads(response_text)

            html_part = json_data.get('collection', [{}])[0].get('html', '')
            if not html_part:
                logging.warning("JSON 응답에 HTML 부분이 없습니다.")
                return [], None
            
            soup = BeautifulSoup(html_part, 'lxml')
            # 2페이지 전용 파서를 호출
            articles = self._parse_article_li_from_json_html(soup)
            
            if not articles:
                logging.error("JSON 내부 HTML에서 게시글을 파싱하는데 실패했습니다. 'debug_page_2.html'에 저장합니다.")
                debug_file_path = os.path.join(self.base_dir, 'debug_page_2.html')
                with open(debug_file_path, 'w', encoding='utf-8') as f:
                    f.write(html_part)
            
            next_page_url = json_data.get('url')
            if next_page_url:
                if not next_page_url.startswith('http'):
                    next_page_url = "https://s.search.naver.com" + next_page_url
                logging.info(f"다음 페이지 API URL 발견 (JSON): {next_page_url}")

            return articles, next_page_url
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            logging.error(f"JSON 파싱 중 오류 발생: {e}")
            # 디버깅을 위해 원본 데이터가 문자열일 경우 기록
            if isinstance(response_text, str):
                 logging.error(f"파싱 실패한 내용: {response_text[:500]}...")
            return [], None

    async def _process_article_batch(self, session, articles, year, month, page):
        """게시글 묶음을 처리합니다. (LLM 필터링 비활성화, 1단계 디버깅용)"""
        if not articles:
            logging.warning(f"{year}-{month} {page}페이지에서 게시글을 찾을 수 없습니다.")
            return

        # 1단계 디버깅을 위해 LLM 필터링 및 상세 정보 수집 단계를 건너뛰고,
        # 초기 파싱 결과물을 즉시 저장합니다.
        logging.info(f"[{year}-{month} {page}페이지] 1단계 수집 결과 {len(articles)}개를 바로 저장합니다. (LLM 필터링 생략)")

        output_dir = os.path.join(os.path.dirname(__file__), 'data', 'historical_raw')
        os.makedirs(output_dir, exist_ok=True)
        # 파일명을 바꿔서 기존 결과와 겹치지 않게 합니다.
        filename = os.path.join(output_dir, f"{year}-{month}_page_{page}_step1_raw_test.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
            logging.info(f"1단계 수집 결과 저장 완료: {filename}")
        except Exception as e:
            logging.error(f"파일 저장 중 오류 발생 ({filename}): {e}")
            
    async def _filter_and_enrich(self, session, article):
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
                    return None # 최종 실패 시 None 반환
                await asyncio.sleep(1)

            except Exception as e:
                logging.error(f"LLM/상세정보 처리 중 예상치 못한 오류: {e}", exc_info=True)
                article['is_related_analysis'] = {"error": str(e)}
                return None # 예외 발생 시 None 반환
        
        return None # 모든 재시도 실패 시

    async def _fetch_details_async(self, session, article):
        """게시글의 상세 정보(조회수, 댓글 등)를 비동기적으로 가져옵니다."""
        if 'club_id' not in article or not article['club_id']:
            return article # club_id가 없으면 반환

        max_retries = 3
        # 프록시 리스트 복사본을 만들어, 실패 시 안전하게 제거
        available_proxies = self.proxies[:] if self.proxies else [None]

        for attempt in range(max_retries):
            # 매 시도마다 다른 프록시 사용 (사용 가능한 프록시가 있을 경우)
            proxy = f"http://{random.choice(available_proxies)}" if available_proxies else None

            try:
                # 1. 댓글 및 조회수 정보 가져오기 (v3 API)
                cafe_url_base = "/".join(article['link'].split('/')[:4])
                if cafe_url_base not in self.club_id_cache:
                    async with session.get(article['link'], timeout=15, proxy=proxy) as response:
                        if response.status == 200:
                            html = await response.text()
                            match = re.search(r'var g_sClubId = "(\d+)"', html)
                            if match:
                                club_id = match.group(1)
                                self.club_id_cache[cafe_url_base] = club_id
                
                article['club_id'] = self.club_id_cache.get(cafe_url_base)
                
                if not article.get('club_id') or not article.get('art_token'):
                    return article # 더 이상 진행 불가

                # 4. 상세 정보 수집
                api_url = f"https://apis.naver.com/cafe-web/cafe-articleapi/v3/cafes/{article['club_id']}/articles/{article['article_id']}?query=&art={article['art_token']}&useCafeId=true&requestFrom=A"
                async with session.get(api_url, timeout=15, proxy=proxy) as response:
                    if response.status == 200:
                        details = await response.json()
                        result_data = details.get('result', {})
                        if 'article' in result_data:
                            article_data = result_data.get('article', {})
                            article['views'] = article_data.get('readCount', 0)
                            comments_data = result_data.get('comments', {}).get('items', [])
                        else:
                            article['views'] = result_data.get('readCount', 0)
                            comments_data = []
                        
                        new_comments = []
                        for c in comments_data:
                            timestamp = c.get('updateDate', 0)
                            # Naver API timestamp는 밀리초 단위이므로 1000으로 나눔
                            formatted_date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y.%m.%d.')
                            new_comments.append({'date': formatted_date, 'content': c.get('content','').strip()})
                        article['comments'] = new_comments
                        return article # 성공 시 루프 탈출
                return article # 성공 시 루프 탈출
            except Exception as e:
                logging.warning(f"상세 정보 수집 실패 (시도 {attempt + 1}/{max_retries}, 프록시: {proxy}): {article['title'][:20]}... - {e}")
                # 실패한 프록시는 리스트에서 제거
                if proxy and proxy.lstrip("http://") in available_proxies:
                    available_proxies.remove(proxy.lstrip("http://"))
                    logging.info(f"API 접근에 실패한 프록시 {proxy}를 풀에서 제거합니다. 남은 API용 프록시: {len(available_proxies)}개")
                
                if not available_proxies:
                    logging.error("사용 가능한 프록시가 모두 소진되어 상세 정보 수집을 중단합니다.")
                    break # 프록시 없으면 더이상 시도하지 않음

                await asyncio.sleep(random.uniform(1, 3))

        logging.error(f"최종 상세 정보 수집 실패: {article['title'][:20]}...")
        return article

    def _save_processed_data(self, year, month, page, articles):
        """처리 완료된 한 페이지의 데이터를 파일로 저장합니다."""
        if not articles: return
        
        file_suffix = "_test" if self.test_mode else ""
        filename = f"{year:04d}-{month:02d}_page_{page:03d}_processed{file_suffix}.json"
        filepath = os.path.join(self.raw_data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
        logging.info(f"{len(articles)}개의 상세 데이터를 '{filepath}'에 저장했습니다.")

    def _log_llm_error(self, article_title, original_response):
        """LLM 오류를 별도의 로그 파일에 기록합니다."""
        log_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, 'llm_errors_test.log' if self.test_mode else 'llm_errors.log')
        
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
            end_date=datetime(2024, 6, 30),
            keyword="동탄 치과",
            test_mode=True
        )
        await crawler.run_async()

    asyncio.run(main()) 
