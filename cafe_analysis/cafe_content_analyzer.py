import os
import json
import logging
import re
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

# --- 로깅 설정 ---
# 이 모듈을 사용하는 스크립트에서 로깅 설정을 하도록 변경합니다.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LLM 에이전트 설정 ---
def get_llm_agent():
    """LLM 에이전트를 초기화하고 반환하는 함수."""
    try:
        # .env 파일 경로는 이 함수를 호출하는 스크립트의 위치를 기준으로 찾습니다.
        load_dotenv(find_dotenv())
        llm = Gemini(id=os.getenv("GEMINI_LITE_MODEL", "gemini-1.5-pro-latest"), api_key=os.getenv("GEMINI_API_KEY"))
        return Agent(model=llm)
    except Exception as e:
        logging.error(f"LLM 에이전트 초기화 실패: {e}")
        return None

def _extract_json(text: str):
    """LLM 응답 텍스트에서 JSON 객체만 추출합니다."""
    # LLM이 코드 블록(```json ... ```)으로 응답하는 경우를 대비하여 정규식 수정
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not match:
        match = re.search(r'(\{.*?\})', text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logging.warning(f"JSON 파싱 실패: {match.group(1)}")
            return {"error": "JSON 파싱 실패"}
    return {"error": "JSON 객체를 찾을 수 없음"}


# --- 분석 클래스 정의 ---
class CafePostAnalyzer:
    """카페 게시글 데이터를 받아 LLM으로 분석하는 클래스."""
    def __init__(self, agent):
        self.agent = agent
        self.prompt_template = self._get_prompt_template()

    def _get_prompt_template(self):
        return """
        당신은 동탄 지역의 치과 시장을 분석하는 마케팅 전문가입니다.
        주어진 네이버 카페 게시글의 '제목', '본문', '댓글'을 종합적으로 분석하여 아래 형식에 맞춰 정보를 추출해주세요.

        **추출할 정보:**
        1.  **mentioned_clinics (list of strings):** 게시글에서 언급된 모든 치과 병원 이름을 정확히 추출해주세요. (예: ["내이튼치과", "서울S치과"])
        2.  **clinic_sentiments (list of objects):** 각 병원에 대한 평판(긍정, 부정, 중립)과 그 이유를 간략하게 추출해주세요. (예: [{{"clinic_name": "내이튼치과", "sentiment": "긍정", "reason": "대표원장이 보존과 전문의이고 필요한 치료만 권한다는 댓글이 있음."}}])
        3.  **main_keywords (list of strings):** 이 게시글에서 드러나는 환자의 핵심적인 관심사나 키워드를 5개 이내로 추출해주세요. (예: ["충치 치료", "과잉진료 없는 곳", "아이 치과", "신경치료 비용", "임플란트 후기"])
        4.  **related_treatments (list of strings):** 게시글 내용과 가장 밀접한 관련이 있는 진료 분야를 아래 **[지정된 목록]** 에서 모두 선택해주세요.

        **[지정된 목록]:**
        ["교정치료", "임플란트", "충치치료", "신경치료", "잇몸치료", "치아보존술", "심미치료", "예방치료(스케일링 등)", "구강외과(사랑니 발치 등)"]

        **분석할 데이터:**
        - 제목: {title}
        - 본문: {preview}
        - 댓글: {comments}

        **지시사항:**
        - 답변은 반드시 ```json ... ``` 코드 블록 안에 유효한 JSON 형식으로만 제공해야 합니다. 다른 어떤 설명도 추가하지 마세요.
        """

    def analyze(self, post_data):
        """단일 게시글 데이터를 받아 분석을 수행하고 결과를 반환합니다."""
        if not self.agent:
            return {"error": "LLM 에이전트가 초기화되지 않았습니다."}

        # 'comments'가 리스트가 아닐 경우를 대비한 방어 코드
        comments = post_data.get('comments', [])
        if not isinstance(comments, list):
            comments = []

        prompt = self.prompt_template.format(
            title=post_data.get('title', ''),
            preview=post_data.get('preview', ''), # 크롤링 데이터의 'preview'는 사실상 본문
            comments=json.dumps(comments, ensure_ascii=False)
        )
        try:
            # agent.run()의 message 키워드 사용
            response = self.agent.run(message=prompt)
            analysis_result = _extract_json(response.content)
            return analysis_result
        except Exception as e:
            logging.error(f"'{post_data.get('title', '')}' 분석 중 LLM 오류: {e}")
            return {"error": "LLM 분석 실패"}
