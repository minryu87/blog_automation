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
OUTPUT_SUBFOLDER = '0728_2300'
CORE_INTEREST_AREAS = ['치아보존술', '임플란트', '충치치료', '신경치료', '잇몸치료', '심미치료']

# --- LLM 에이전트 설정 ---
def get_llm_agent():
    try:
        load_dotenv(find_dotenv())
        llm = Gemini(id=os.getenv("GEMINI_LITE_MODEL", "gemini-2.5-pro"), api_key=os.getenv("GEMINI_API_KEY"))
        return Agent(model=llm)
    except Exception as e:
        logging.error(f"LLM 에이전트 초기화 실패: {e}")
        return None

def _extract_list(text: str):
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group(0))
        except (ValueError, SyntaxError):
            return []
    return []

# --- 1. 규칙 기반 라벨링 및 스테이지 분류 ---
class RuleBasedTagger:
    def __init__(self, taxonomy_path):
        self.taxonomy = self._load_taxonomy(taxonomy_path)
        self.keyword_map = self._build_keyword_map()

    def _load_taxonomy(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"택소노미 파일을 찾을 수 없습니다: {path}")
            return {}

    def _build_keyword_map(self):
        keyword_map = {}
        for category, labels in self.taxonomy.items():
            for label, expressions in labels.items():
                for exp in expressions:
                    keyword_map[exp.lower()] = f"{category}:{label}"
        return keyword_map

    def label_query(self, query):
        query_lower = query.lower()
        matched_results = []
        
        # 긴 키워드 우선 매칭을 위해 키워드 길이로 정렬
        sorted_keywords = sorted(self.keyword_map.keys(), key=len, reverse=True)

        for keyword in sorted_keywords:
            if keyword in query_lower:
                matched_results.append({
                    "matched_expression": keyword,
                    "label": self.keyword_map[keyword]
                })
                query_lower = query_lower.replace(keyword, '') # 중복 방지

        return matched_results

    def assign_stage(self, labels):
        has_location = False
        has_hospital_name = False

        for label_info in labels:
            # label_info 예시: {"matched_expression": "동탄", "label": "지역/장소:동탄"}
            try:
                label_value = label_info.get('label', '')
                if not label_value or ':' not in label_value:
                    continue

                category, label_name = label_value.split(':', 1)

                if category == '지역/장소':
                    has_location = True
                    # 'ㅇㅇ치과' 처럼 라벨 자체에 '치과'가 포함된 경우를 병원 인지로 간주
                    if '치과' in label_name:
                        has_hospital_name = True
            except (ValueError, AttributeError) as e:
                logging.warning(f"라벨 분석 중 오류 발생: {label_info} - {e}")
                continue
        
        if has_hospital_name:
            return '4단계:병원인지'
        elif has_location:
            return '3단계:지역탐색'
        else:
            return '2단계:정보탐색'

# --- 2. LLM 기반 관심 진료 분야 추론 ---
class InterestAreaExtractor:
    def __init__(self, agent):
        self.agent = agent
        self.prompt_template = self._get_prompt_template()

    def _get_prompt_template(self):
        return """
        당신은 환자의 검색 의도를 파악하는 치과 마케팅 분석 전문가입니다.
        주어진 '검색어'와 '1차 분석 라벨'을 종합적으로 고려하여, 이 사용자의 핵심 '관심 진료 분야'가 무엇인지 추론해야 합니다.

        **핵심 관심 진료 분야 리스트:**
        {interest_areas}

        ---
        **사고 과정 (Chain of Thought):**
        1. '검색어'와 '1차 분석 라벨'에 나타난 핵심 증상과 상황을 분석합니다. (예: '치수괴사'는 신경이 죽은 상태를 의미한다.)
        2. 이 증상/상황을 해결하기 위해 필요한 치과 진료가 무엇인지 생각합니다. (예: 죽은 신경을 제거하는 것은 '신경치료'에 해당한다.)
        3. 이 진료가 '핵심 관심 진료 분야 리스트'에 있는지 확인하고, 관련 있는 모든 항목을 선택합니다.

        ---
        **예시:**

        *   **입력:**
            *   검색어: "동탄 지르코니아 크라운"
            *   1차 분석 라벨: [{{"matched_expression": "동탄", "label": "지역/장소:지역명"}}, {{"matched_expression": "지르코니아", "label": "의료/진료방식:재료"}}, {{"matched_expression": "크라운", "label": "의료/진료방식:보철"}}]
        *   **사고 과정:**
            1. '크라운' 치료는 보통 '신경치료'를 마친 치아를 보호하거나, '임플란트' 상부 보철물로 사용되거나, 심미적인 개선을 위해 사용된다.
            2. '지르코니아'는 심미성이 좋은 재료이다.
            3. 따라서, 이 검색어는 '신경치료', '임플란트', '심미치료'와 모두 관련될 수 있다.
        *   **결과:** ["신경치료", "임플란트", "심미치료"]

        *   **입력:**
            *   검색어: "교정 치수괴사"
            *   1차 분석 라벨: [{{"matched_expression": "치수괴사", "label": "증상/상태:염증"}}, {{"matched_expression": "교정", "label": "증상/상태:교정 중 증상"}}]
        *   **사고 과정:**
            1. '치수괴사'는 치아 신경이 죽은 상태이다.
            2. 죽은 신경을 제거하고 치료하는 것은 '신경치료'이다.
            3. 신경이 죽은 치아는 시간이 지나며 검게 변색될 수 있어 '심미치료'(예: 실활치 미백)가 필요할 수 있다.
        *   **결과:** ["신경치료", "심미치료"]
        ---

        **분석할 데이터:**
        - 검색어: "{query}"
        - 1차 분석 라벨: {labels}
        ---

        **지시사항:**
        1. 위의 사고 과정과 예시를 참고하여, 주어진 데이터에 대한 '핵심 관심 진료 분야'를 추론하세요.
        2. 관련성이 명확하지 않거나 없다고 판단될 경우에만 빈 리스트를 반환하세요.
        3. 답변은 반드시 파이썬 리스트(list) 형식으로만 제공해야 합니다. (예: ["임플란트", "잇몸치료"] 또는 [])
        4. 다른 어떤 설명도 추가하지 마세요.
        """

    def extract(self, query, labels):
        if not self.agent:
            return []
        
        prompt = self.prompt_template.format(
            interest_areas=json.dumps(CORE_INTEREST_AREAS, ensure_ascii=False),
            query=query,
            labels=json.dumps(labels, ensure_ascii=False)
        )
        try:
            response = self.agent.run(prompt)
            return _extract_list(response.content)
        except Exception as e:
            logging.error(f"'{query}' 처리 중 LLM 오류 발생: {e}")
            return ["LLM Error"]

# --- 메인 실행 로직 ---
def main():
    logging.info("===== v2 택소노미 적용 및 관심 진료 분야 추론 시스템 시작 =====")

    # --- 경로 설정 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    input_queries_path = os.path.join(project_root, 'blog_automation', 'data', 'data_input', 'searchQuery.csv')
    taxonomy_path = os.path.join(script_dir, 'taxonomy_v2.json')
    
    output_dir = os.path.join(script_dir, 'data', 'data_processed', OUTPUT_SUBFOLDER)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'final_labeled_queries.csv')
    
    # --- 초기화 ---
    tagger = RuleBasedTagger(taxonomy_path)
    agent = get_llm_agent()
    extractor = InterestAreaExtractor(agent)

    queries_df = pd.read_csv(input_queries_path)
    
    # --- 이어하기 로직 ---
    processed_queries = set()
    output_file_exists = os.path.exists(output_path)
    if output_file_exists:
        try:
            processed_df = pd.read_csv(output_path)
            processed_queries = set(processed_df['searchQuery'])
            logging.info(f"기존 결과 파일에서 {len(processed_queries)}개의 처리된 검색어를 발견했습니다. 이어서 작업을 시작합니다.")
        except (pd.errors.EmptyDataError, FileNotFoundError):
            output_file_exists = False
        except Exception as e:
            logging.error(f"기존 결과 파일 읽기 오류: {e}. 처음부터 시작합니다.")
            output_file_exists = False
    
    # --- END ---

    pbar = tqdm(total=queries_df.shape[0], desc="검색어 처리 중", initial=len(processed_queries))
    
    for _, row in queries_df.iterrows():
        query = row.iloc[0]
        
        if query in processed_queries:
            continue

        pbar.update(1)

        if not keep_running:
            logging.info("작업 중단. 현재까지의 결과를 저장합니다.")
            break
            
        # 1 & 2단계
        labels = tagger.label_query(query)
        stage = tagger.assign_stage(labels)
        
        # 3단계
        interest_areas = extractor.extract(query, labels)
        
        result_entry = pd.DataFrame([{
            "searchQuery": query,
            "labels(json)": json.dumps(labels, ensure_ascii=False),
            "stage": stage,
            "interest_areas": json.dumps(interest_areas, ensure_ascii=False)
        }])
        
        result_entry.to_csv(output_path, mode='a', index=False, header=(not output_file_exists), encoding='utf-8-sig')
        output_file_exists = True

    pbar.close()
    
    logging.info(f"모든 작업 완료. 최종 결과는 다음 파일에 저장되었습니다: {output_path}")

if __name__ == "__main__":
    main() 