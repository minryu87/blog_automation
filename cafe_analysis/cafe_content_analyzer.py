import os
import json
import pandas as pd
import logging
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
import ast
import re
import signal
import sys
from datetime import datetime, timedelta

# --- 로깅 설정 및 시그널 핸들러 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
keep_running = True

def signal_handler(sig, frame):
    global keep_running
    if keep_running:
        logging.warning("\n[Ctrl+C 감지] 현재 작업을 완료한 후 저장을 시도합니다...")
        keep_running = False
    else:
        logging.warning("\n[강제 종료] 즉시 프로그램을 종료합니다.")
        sys.exit(1)
signal.signal(signal.SIGINT, signal_handler)

# --- 상수 정의 ---
TODAY_STR = datetime.now().strftime('%m%d_%H%M')
OUTPUT_SUBFOLDER = TODAY_STR

# --- LLM 에이전트 설정 (apply_restructured_taxonomy.py 참고) ---
def get_llm_agent():
    try:
        load_dotenv(find_dotenv())
        # 모델 ID를 최신 버전으로 명시
        llm = Gemini(id=os.getenv("GEMINI_LITE_MODEL", "gemini-1.5-pro-latest"), api_key=os.getenv("GEMINI_API_KEY"))
        return Agent(model=llm)
    except Exception as e:
        logging.error(f"LLM 에이전트 초기화 실패: {e}")
        return None

def _extract_json(text: str):
    """LLM 응답 텍스트에서 JSON 객체만 추출합니다."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logging.warning(f"JSON 파싱 실패: {match.group(0)}")
            return {"error": "JSON 파싱 실패"}
    return {"error": "JSON 객체를 찾을 수 없음"}


# --- 분석 클래스 정의 ---
class CafePostAnalyzer:
    def __init__(self, agent):
        self.agent = agent
        self.prompt_template = self._get_prompt_template()

    def _get_prompt_template(self):
        # TODO: LLM에게 지시할 프롬프트 작성
        return """
        당신은 동탄 지역의 치과 시장을 분석하는 마케팅 전문가입니다.
        주어진 네이버 카페 게시글의 '제목', '본문', '댓글'을 종합적으로 분석하여 아래 형식에 맞춰 정보를 추출해주세요.

        1.  **is_related_to_dongtan_dental (boolean):** 이 게시글이 '동탄 지역 치과'와 직접적으로 관련된 내용이 맞습니까? (예: 동탄 지역 내 치과 추천, 문의, 후기 등)
        2.  **mentioned_clinics (list of strings):** 게시글에서 언급된 모든 치과 병원 이름을 정확히 추출해주세요. (예: ["내이튼치과", "서울S치과"])
        3.  **clinic_sentiments (list of objects):** 각 병원에 대한 평판(긍정, 부정, 중립)과 그 이유를 간략하게 추출해주세요. (예: [{{"clinic_name": "내이튼치과", "sentiment": "긍정", "reason": "대표원장이 보존과 전문의이고 필요한 치료만 권한다는 댓글이 있음."}}])
        4.  **main_keywords (list of strings):** 이 게시글에서 드러나는 환자의 핵심적인 관심사나 키워드를 5개 이내로 추출해주세요. (예: ["충치 치료", "과잉진료 없는 곳", "아이 치과", "신경치료 비용", "임플란트 후기"])

        **분석할 데이터:**
        - 제목: {title}
        - 본문: {preview}
        - 댓글: {comments}

        **지시사항:**
        - 답변은 반드시 유효한 JSON 형식으로만 제공해야 합니다. 다른 어떤 설명도 추가하지 마세요.
        - `is_related_to_dongtan_dental`이 false일 경우, 다른 필드는 모두 비워두세요.
        """

    def analyze(self, post_data):
        if not self.agent:
            return {"error": "LLM 에이전트가 초기화되지 않았습니다."}

        # LLM 호출 로직
        prompt = self.prompt_template.format(
            title=post_data.get('title', ''),
            preview=post_data.get('preview', ''),
            comments=json.dumps(post_data.get('comments', []), ensure_ascii=False)
        )
        try:
            response = self.agent.run(prompt)
            analysis_result = _extract_json(response.content)
            return analysis_result
        except Exception as e:
            logging.error(f"'{post_data.get('title', '')}' 분석 중 LLM 오류: {e}")
            return {"error": "LLM 분석 실패"}

def calculate_view_increase(today_output_dir, processed_posts):
    """어제 데이터와 비교하여 일일 조회수 증가량을 계산하고 CSV 파일로 저장합니다."""
    logging.info("--- 일일 조회수 증가량 분석 시작 ---")
    
    today_df = pd.DataFrame(processed_posts)
    # 'views' 컬럼이 문자열일 경우 숫자형으로 변환 (오류 발생 시 NaN으로 처리)
    today_df['views'] = pd.to_numeric(today_df['views'], errors='coerce').fillna(0).astype(int)

    # 어제 날짜 폴더 경로 탐색
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%m%d')
    processed_dir = os.path.dirname(today_output_dir)
    yesterday_dir = None
    
    # 어제 날짜와 일치하는 폴더를 찾음 (예: 0728_*)
    try:
        for dirname in os.listdir(processed_dir):
            if dirname.startswith(yesterday_str):
                yesterday_dir_path = os.path.join(processed_dir, dirname)
                if os.path.isdir(yesterday_dir_path):
                    yesterday_dir = yesterday_dir_path
                    break
    except FileNotFoundError:
        logging.warning("processed 폴더를 찾을 수 없어 어제 데이터를 로드할 수 없습니다.")
        yesterday_dir = None


    if not yesterday_dir:
        logging.warning("어제 분석 데이터가 없어 조회수 증가량을 계산할 수 없습니다. 오늘 조회수를 기준으로 파일을 생성합니다.")
        today_df['yesterday_views'] = 0
        today_df['view_increase'] = today_df['views']
    else:
        yesterday_file_path = os.path.join(yesterday_dir, 'analyzed_cafe_posts.json')
        try:
            yesterday_df = pd.read_json(yesterday_file_path)
            yesterday_df['views'] = pd.to_numeric(yesterday_df['views'], errors='coerce').fillna(0).astype(int)
            
            # 데이터를 article_id 기준으로 병합
            merged_df = pd.merge(
                today_df[['article_id', 'views']],
                yesterday_df[['article_id', 'views']],
                on='article_id',
                how='left',
                suffixes=('_today', '_yesterday')
            )
            merged_df['views_yesterday'] = merged_df['views_yesterday'].fillna(0).astype(int)
            
            # 오늘 데이터프레임에 어제 조회수와 증가량 추가
            today_df = pd.merge(today_df, merged_df[['article_id', 'views_yesterday']], on='article_id', how='left')
            today_df.rename(columns={'views_yesterday': 'yesterday_views'}, inplace=True)
            today_df['yesterday_views'] = today_df['yesterday_views'].fillna(0).astype(int)
            today_df['view_increase'] = today_df['views'] - today_df['yesterday_views']

        except FileNotFoundError:
            logging.warning(f"어제 분석 파일({yesterday_file_path})을 찾을 수 없습니다. 오늘 조회수를 기준으로 파일을 생성합니다.")
            today_df['yesterday_views'] = 0
            today_df['view_increase'] = today_df['views']

    # 필요한 컬럼만 선택하여 CSV로 저장
    output_columns = [
        'article_id', 'title', 'link', 'yesterday_views', 'views', 'view_increase', 
        'is_related_to_dongtan_dental', 'mentioned_clinics', 'clinic_sentiments', 'main_keywords'
    ]
    # 'views' 컬럼명을 'today_views'로 변경
    today_df.rename(columns={'views': 'today_views'}, inplace=True)
    output_columns[4] = 'today_views'

    # 최종 데이터프레임에 모든 필요한 컬럼이 있는지 확인하고 없으면 빈 값으로 채움
    for col in output_columns:
        if col not in today_df.columns:
            today_df[col] = None

    output_csv_path = os.path.join(today_output_dir, 'daily_view_counts.csv')
    today_df[output_columns].to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"일일 조회수 분석 완료. 결과는 다음 파일에 저장되었습니다: {output_csv_path}")


# --- 메인 실행 로직 ---
def main():
    logging.info("===== 네이버 카페 게시글 LLM 분석 시스템 시작 =====")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(script_dir, 'data', 'data_input', 'naver_cafe_crawl_result.json')
    
    output_dir = os.path.join(script_dir, 'data', 'data_processed', OUTPUT_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'analyzed_cafe_posts.json')
    daily_views_path = os.path.join(output_dir, 'daily_view_counts.csv')

    # --- 초기화 ---
    agent = get_llm_agent()
    if not agent:
        logging.error("LLM 에이전트를 사용할 수 없어 프로그램을 종료합니다.")
        return
        
    analyzer = CafePostAnalyzer(agent)

    try:
        # 이 부분은 사용자가 직접 파일을 복사했다고 가정합니다.
        with open(input_path, 'r', encoding='utf-8') as f:
            crawled_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        logging.error("ai-api/naver_cafe_crawl_result_v2.json 파일을 blog_automation/cafe_analysis/data/data_input/naver_cafe_crawl_result.json 으로 복사해주세요.")
        return

    all_posts = [post for posts in crawled_data.values() for post in posts]
    
    final_results = []
    pbar = tqdm(all_posts, desc="카페 게시글 분석 중")

    for post in pbar:
        if not keep_running:
            break
        analysis_result = analyzer.analyze(post)
        post.update(analysis_result)
        final_results.append(post)

    pbar.close()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
        
    logging.info(f"분석 완료. 결과는 다음 파일에 저장되었습니다: {output_path}")
    
    # 일일 조회수 변화량 계산 및 저장 로직 추가
    calculate_view_increase(output_dir, final_results)

if __name__ == "__main__":
    main()
