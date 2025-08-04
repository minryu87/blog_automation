import os
import json
import logging
import asyncio
import glob
from datetime import datetime
from collections import OrderedDict

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

# --- 로깅 설정 ---
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

class FinalAnalyzer:
    def __init__(self):
        # 경로 설정
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.raw_data_dir = os.path.join(self.base_dir, 'data', 'historical_raw')
        self.processed_data_dir = os.path.join(self.base_dir, 'data', 'historical_processed')
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # LLM 에이전트 초기화
        load_dotenv(find_dotenv())
        try:
            llm = Gemini(id=os.getenv("GEMINI_LITE_MODEL", "gemini-1.5-pro-latest"), api_key=os.getenv("GEMINI_API_KEY"))
            self.agent = Agent(model=llm)
        except Exception as e:
            logging.error(f"LLM 에이전트 초기화 실패: {e}")
            self.agent = None

        self.analysis_prompt_template = """
        당신은 대한민국 '동탄' 지역의 치과 시장을 분석하는 마케팅 전문가입니다.
        주어진 네이버 카페 게시글의 '제목', '본문 미리보기', 그리고 '전체 댓글' 내용을 종합적으로 분석하여, 다음 정보를 JSON 형식으로만 추출해주세요.

        **분석 항목:**
        1.  `mentioned_clinics` (list of strings): 게시글 본문 및 댓글에서 언급된 '동탄'에 위치한 모든 치과 병원의 이름을 정확히 추출해주세요. '동탄'이라는 키워드가 없더라도 문맥상 동탄 지역임이 확실하면 포함합니다. (예: ["동탄퍼스트치과", "연세본치과"])
        2.  `clinic_sentiments` (list of objects): 언급된 각 병원에 대한 구체적인 평판(긍정, 부정, 중립)과 그 핵심 이유를 간략하게 요약해주세요. 긍정/부정의 근거가 되는 특정 단어나 문장을 'reason'에 포함시키세요. (예: [{"clinic_name": "동탄퍼스트치과", "sentiment": "긍정", "reason": "'과잉진료 없고 설명이 친절하다'는 댓글 내용"}])
        3.  `main_keywords` (list of strings): 이 게시글에서 환자들이 가장 궁금해하거나 중요하게 생각하는 핵심적인 관심사, 시술, 또는 증상 키워드를 5개 이내로 추출해주세요. (예: ["임플란트 비용", "과잉진료 없는 치과", "어린이 충치치료", "신경치료 후기", "사랑니 발치"])
        4.  `is_advertising` (boolean): 게시글 또는 댓글 내용이 명백한 광고/홍보성으로 판단되는 경우 true, 일반적인 후기나 질문이면 false로 지정해주세요.

        **분석할 데이터:**
        - 제목: {title}
        - 본문 미리보기: {preview}
        - 전체 댓글: {comments}

        **지시사항:**
        - 답변은 반드시 유효한 JSON 객체 하나만 반환해야 하며, 다른 어떤 설명도 추가해서는 안 됩니다.
        - 언급된 치과가 없으면 `mentioned_clinics`와 `clinic_sentiments`는 빈 리스트 `[]`로 반환하세요.
        - 평판을 판단할 근거가 부족하면 `sentiment`는 '중립'으로 지정하세요.
        """

    def _fix_mentioned_clinics(self, articles):
        """
        clinic_sentiments의 clinic_name들을 mentioned_clinics에 추가하여 중복 없는 리스트로 만듭니다.
        """
        for article in articles:
            analysis = article.get('analysis', {})
            if not analysis:
                continue
            
            mentioned_clinics = analysis.get('mentioned_clinics', [])
            clinic_sentiments = analysis.get('clinic_sentiments', [])
            
            # clinic_sentiments에서 clinic_name 추출
            sentiment_clinics = []
            for sentiment in clinic_sentiments:
                clinic_name = sentiment.get('clinic_name')
                if clinic_name:
                    sentiment_clinics.append(clinic_name)
            
            # 모든 clinic_name을 합쳐서 중복 제거
            all_clinics = mentioned_clinics + sentiment_clinics
            unique_clinics = list(OrderedDict.fromkeys(all_clinics))  # 순서 유지하면서 중복 제거
            
            # mentioned_clinics 업데이트
            analysis['mentioned_clinics'] = unique_clinics
        
        return articles

    def _load_all_processed_data(self):
        """historical_raw 폴더의 모든 _processed.json 파일을 로드하여 합칩니다."""
        json_files = glob.glob(os.path.join(self.raw_data_dir, '*_processed.json'))
        if not json_files:
            logging.warning("처리할 중간 결과 파일이 없습니다.")
            return []
            
        all_articles = []
        logging.info(f"{len(json_files)}개의 중간 결과 파일을 로드합니다.")
        for file_path in tqdm(json_files, desc="중간 결과 파일 로딩 중"):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    all_articles.extend(data)
                except json.JSONDecodeError:
                    logging.error(f"JSON 파싱 오류: {file_path}")
        
        # 중복 제거 (동일한 article_id가 여러 파일에 걸쳐 있을 경우 대비)
        df = pd.DataFrame(all_articles)
        df.drop_duplicates(subset='article_id', keep='last', inplace=True)
        
        logging.info(f"총 {len(df)}개의 고유한 게시글을 로드했습니다.")
        return df.to_dict('records')

    async def _analyze_article_detail(self, article):
        """하나의 게시글을 받아 LLM으로 심층 분석하고 결과를 반환합니다."""
        if not self.agent: return {"error": "LLM agent not initialized."}

        prompt = self.analysis_prompt_template.format(
            title=article.get('title', ''),
            preview=article.get('preview', ''),
            comments=json.dumps(article.get('comments', []), ensure_ascii=False)
        )
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.agent.run, prompt)
            analysis_result = json.loads(response.content)
            article['analysis'] = analysis_result
        except Exception as e:
            logging.error(f"LLM 심층 분석 오류 ({article.get('title', '')[:20]}...): {e}")
            article['analysis'] = {"error": str(e)}
        return article

    async def run_analysis(self):
        """전체 분석 파이프라인을 실행합니다."""
        logging.info("===== 최종 심층 분석 시작 =====")
        
        articles = self._load_all_processed_data()
        if not articles:
            logging.info("분석할 데이터가 없어 종료합니다.")
            return

        logging.info(f"{len(articles)}개 게시글에 대한 심층 분석을 시작합니다.")
        
        analysis_tasks = [self._analyze_article_detail(art) for art in articles]
        
        # tqdm을 비동기 작업에 적용하기 위한 처리
        results = []
        for f in tqdm.as_completed([asyncio.ensure_future(task) for task in analysis_tasks], total=len(analysis_tasks), desc="심층 분석 진행 중"):
            results.append(await f)

        # mentioned_clinics 수정
        logging.info("mentioned_clinics 수정 작업을 시작합니다.")
        results = self._fix_mentioned_clinics(results)
        logging.info("mentioned_clinics 수정 작업이 완료되었습니다.")

        final_df = pd.DataFrame(results)
        
        # 최종 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(self.processed_data_dir, f'final_analysis_{timestamp}.json')
        csv_path = os.path.join(self.processed_data_dir, f'final_analysis_{timestamp}.csv')
        
        # analysis 컬럼(dict)을 분해하여 별도의 컬럼으로 추가 (CSV 저장을 위해)
        if 'analysis' in final_df.columns:
            analysis_df = pd.json_normalize(final_df['analysis'])
            final_df = final_df.join(analysis_df)
        
        final_df.to_json(json_path, orient='records', lines=True, force_ascii=False)
        final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logging.info(f"분석 완료. 결과가 다음 파일에 저장되었습니다:\n- {json_path}\n- {csv_path}")

if __name__ == '__main__':
    analyzer = FinalAnalyzer()
    asyncio.run(analyzer.run_analysis()) 