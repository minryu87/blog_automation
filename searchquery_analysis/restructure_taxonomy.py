import os
import json
import pandas as pd
import logging
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
import ast
from collections import defaultdict
import signal
import sys
import re
import random

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 우아한 종료를 위한 글로벌 플래그 및 핸들러 ---
keep_running = True
def signal_handler(sig, frame):
    """Ctrl+C 입력을 감지하여 안전한 종료를 준비합니다."""
    global keep_running
    if keep_running:
        logging.warning("\n[Ctrl+C 감지] 현재 라벨 처리를 완료한 후 저장을 시도합니다. 즉시 종료하려면 다시 한번 눌러주세요.")
        keep_running = False
    else:
        logging.warning("\n[강제 종료] 즉시 프로그램을 종료합니다.")
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)


# --- 새로운 택소노미 카테고리 정의 (LLM에게 제공할 가이드) ---
NEW_TAXONOMY_GUIDE = """
1.  **`고객특성` (Demographics)**: 검색 주체의 인구통계학적 특성. (예: 20대, 여성, 학생, 임산부)
2.  **`시간/시기` (Timing)**: 검색의 시간적 맥락. (예: 응급, 야간, 치료 후, 재발, 만성)
3.  **`관심진료` (Service of Interest)**: 고객이 관심을 보이는 핵심 진료 항목. (예: 임플란트, 신경치료, 충치치료, 치아미백, 스케일링, 브릿지)
4.  **`증상/상태` (Symptoms & Conditions)**: 고객이 인지하는 구체적인 증상이나 상태. (예: 통증, 시림, 깨짐, 금감, 붓기, 고름, 변색, 벌어짐)
5.  **`신체부위` (Anatomy)**: 문제나 관심이 발생한 구체적인 신체 부위. (예: 앞니, 어금니, 잇몸, 치조골, 턱관절)
6.  **`지역/장소` (Location)**: 검색과 관련된 지리적 위치. (예: 동탄, 화성, 동탄역, 내이튼치과)
7.  **`의료/진료방식` (Medical Approach)**: 고객이 궁금해하는 특정 의료 기술, 재료, 장비. (예: 미세현미경, CT, 지르코니아, 골드, 수면마취)
8.  **`검색의도` (Core Intent)**: 검색을 통해 진짜로 알고 싶은 핵심 동기. (예: 비용, 가격, 보험, 후기, 잘하는 곳, 추천, 방법, 과정, 기간, 부작용)
"""

# --- LLM 에이전트 설정 ---
def get_llm_agent():
    """Gemini LLM 에이전트를 초기화하고 반환합니다."""
    try:
        load_dotenv(find_dotenv())
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        llm = Gemini(
            id=os.getenv("GEMINI_LITE_MODEL", "gemini-1.5-pro-latest"),
            api_key=os.getenv("GEMINI_API_KEY")
        )
        return Agent(model=llm)
    except Exception as e:
        logging.error(f"LLM 에이전트 초기화 실패: {e}")
        return None

def _extract_json(text: str):
    """LLM 응답에서 JSON 블록을 안정적으로 추출합니다."""
    # 때때로 LLM이 markdown 코드 블록을 포함하여 반환하는 경우가 있음
    # 예: ```json\n{...}\n```
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Markdown 블록이 없는 경우, 전체 텍스트에서 JSON 객체를 찾음
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logging.warning(f"JSON 파싱 실패: {json_str}")
        return None

def build_context_map(labeled_queries_path):
    """
    라벨링된 쿼리 데이터를 기반으로, 각 라벨이 어떤 검색어에서 등장했는지
    컨텍스트 맵을 구축합니다.
    """
    logging.info("컨텍스트 맵 구축 시작...")
    try:
        df = pd.read_csv(labeled_queries_path)
        df.dropna(subset=['labels(json)', 'searchQuery'], inplace=True)

        context_map = defaultdict(list)
        for _, row in df.iterrows():
            query = row['searchQuery']
            try:
                labels_json = json.loads(row['labels(json)'])
                # 'Category:Label' 형태의 전체 라벨명 추출
                labels = set(item['label'] for item in labels_json)
                for label in labels:
                    context_map[label].append(query)
            except (json.JSONDecodeError, TypeError):
                continue
        
        # 각 라벨별로 예시 쿼리를 최대 5개로 제한 (무작위 샘플링)
        for label, queries in context_map.items():
            if len(queries) > 5:
                context_map[label] = random.sample(queries, 5)
        
        logging.info(f"컨텍스트 맵 구축 완료. {len(context_map)}개의 고유 라벨에 대한 컨텍스트를 확보했습니다.")
        return context_map
    except FileNotFoundError:
        logging.error(f"컨텍스트 맵 구축을 위한 파일을 찾을 수 없습니다: {labeled_queries_path}")
        return None

# --- 메인 재구성 로직 ---
def restructure_taxonomy(mapping_df, agent, log_path, context_map):
    """
    LLM을 사용하여 기존 택소노미를 새로운 가이드에 맞게 재구성합니다.
    """
    logging.info("택소노미 재구성 시작...")
    
    if not agent:
        logging.error("LLM 에이전트가 없어 재구성을 진행할 수 없습니다.")
        return None, None

    # --- 유효성 검사를 위한 카테고리 목록을 정확하게 추출 ---
    valid_categories = []
    for line in NEW_TAXONOMY_GUIDE.strip().split('\n'):
        match = re.search(r'`([^`]+)`', line)
        if match:
            valid_categories.append(match.group(1))
    
    # LLM이 합리적으로 판단하는 '기타' 카테고리를 허용 목록에 추가
    valid_categories.append('기타')
    logging.info(f"유효한 카테고리 목록: {valid_categories}")
    # --- END ---

    prompt_template = """
    당신은 데이터 분석가이자 치과 도메인 전문가입니다.
    기존의 '오래된 라벨'을 아래의 '새로운 택소노미 가이드'에 따라 가장 적합한 새로운 카테고리와 새로운 라벨명으로 재분류해야 합니다.
    '관련 표현들'과 **특히 '사용된 검색어 예시'**를 통해 이 라벨이 사용된 **문맥**을 정확히 파악하세요.

    ---
    **새로운 택소노미 가이드:**
    {taxonomy_guide}
    ---
    **재분류할 정보:**
    - 오래된 카테고리: "{old_category}"
    - 오래된 라벨: "{old_label}"
    - 관련 표현들: "{expressions}"
    - **사용된 검색어 예시 (문맥 파악에 가장 중요):** "{context_examples}"
    ---

    **출력 형식 (JSON 객체):**
    다음과 같은 형식의 JSON 객체 하나만 반환해주세요.
    {{
      "thought": "판단 과정에 대한 간단한 생각.",
      "new_category": "새로운 택소노미 가이드에 명시된 8개 카테고리 중 하나.",
      "new_label": "관련 표현들을 가장 잘 대표하는 새로운 라벨명."
    }}

    **규칙:**
    - '브리짓'과 같은 오타는 '브릿지'처럼 올바른 표현으로 교정하여 new_label을 만드세요.
    - '치수염', '치주염'과 같은 의학 용어는 '염증'처럼 더 일반적이거나 분석에 용이한 대표 용어로 new_label을 만드세요.
    - '세게', '육안' 처럼 분석 가치가 낮은 라벨은 '기타' 또는 다른 적절한 라벨에 통합하는 것을 고려하세요.
    - new_category는 반드시 가이드에 있는 8개 중 하나여야 합니다.
    """

    # --- 이어하기 로직 ---
    processed_labels = set()
    log_file_exists = os.path.exists(log_path)
    if log_file_exists:
        try:
            log_df_existing = pd.read_csv(log_path)
            # 'old_category:old_label' 형태로 조합하여 처리된 라벨 식별
            processed_labels = set(log_df_existing['old_category'] + ':' + log_df_existing['old_label'])
            logging.info(f"로그 파일에서 {len(processed_labels)}개의 처리된 라벨을 발견했습니다. 이어서 작업을 시작합니다.")
        except (pd.errors.EmptyDataError, FileNotFoundError):
            log_file_exists = False # 파일이 비어있으면 처음부터 시작
            logging.info("기존 로그 파일이 비어있거나 찾을 수 없어 처음부터 시작합니다.")
        except Exception as e:
            logging.error(f"로그 파일 읽기 오류: {e}. 처음부터 시작합니다.")
            log_file_exists = False
    
    # --- END ---

    # tqdm의 초기값 설정
    pbar = tqdm(total=mapping_df.shape[0], desc="라벨 재분류 중", initial=len(processed_labels))

    reclassification_log = [] # 이 로그는 최종 결과를 위해 사용되지 않고, 파일에 즉시 추가됨
    new_taxonomy = defaultdict(lambda: defaultdict(list))

    for _, row in mapping_df.iterrows():
        old_category = row['Category']
        old_label = row['Label']
        
        # 이미 처리된 라벨은 건너뛰기
        if f"{old_category}:{old_label}" in processed_labels:
            continue

        pbar.update(1) # tqdm 진행률 수동 업데이트

        if not keep_running:
            logging.info("종료 신호를 감지하여 재분류를 중단합니다. 지금까지의 결과를 저장합니다.")
            break

        expressions = row['Matched Expressions']
        
        # 컨텍스트 맵에서 예시 쿼리 가져오기
        full_old_label = f"{old_category}:{old_label}"
        context_examples = context_map.get(full_old_label, ["N/A"])

        # '관심진료' 관련 라벨은 직접 매핑하여 LLM 호출을 줄임
        if old_category == '진료' and old_label in ['임플란트', '신경치료', '충치', '라미네이트', '스케일링', '브릿지', '잇몸치료']:
             # 간단한 규칙 기반 매핑
            new_cat = '관심진료'
            # '브릿지' 오타 수정
            new_lab = '브릿지' if old_label == '브릿지' else old_label
            thought = "핵심 진료 항목으로 직접 매핑"
        else:
            try:
                prompt = prompt_template.format(
                    taxonomy_guide=NEW_TAXONOMY_GUIDE,
                    old_category=old_category,
                    old_label=old_label,
                    expressions=expressions,
                    context_examples=json.dumps(context_examples, ensure_ascii=False)
                )
                response = agent.run(prompt)
                result = _extract_json(response.content)

                if result:
                    new_cat = result.get("new_category")
                    new_lab = result.get("new_label")
                    thought = result.get("thought")

                    # LLM이 반환한 값의 양쪽 공백, 백틱, 따옴표를 모두 제거
                    if isinstance(new_cat, str):
                        new_cat = new_cat.strip().strip('`\'"')
                    
                    # 최종 유효성 검사
                    if not new_cat or not new_lab or new_cat not in valid_categories:
                        raise ValueError(f"LLM의 응답값이 유효하지 않습니다: {result}")
                else:
                    raise ValueError(f"LLM 응답에서 JSON을 추출할 수 없습니다: {response.content}")

            except Exception as e:
                logging.warning(f"'{old_category}:{old_label}' 처리 중 오류 발생: {e}. '기타:미분류'로 처리합니다.")
                new_cat, new_lab, thought = "기타", "미분류", f"오류 발생: {e}"
        
        # 로그를 DataFrame으로 만들어 파일에 즉시 추가
        log_entry = pd.DataFrame([{
            "old_category": old_category, "old_label": old_label,
            "new_category": new_cat, "new_label": new_lab,
            "expressions": expressions, "thought": thought
        }])
        
        log_entry.to_csv(log_path, mode='a', index=False, header=(not log_file_exists), encoding='utf-8-sig')
        log_file_exists = True # 첫 쓰기 이후에는 항상 파일이 존재함

    pbar.close() # tqdm 진행 바 닫기
    
    # 최종적으로 전체 로그 파일을 읽어 새로운 택소노미 구성
    if os.path.exists(log_path):
        final_log_df = pd.read_csv(log_path)
        new_taxonomy = defaultdict(lambda: defaultdict(list))
        for _, row in final_log_df.iterrows():
            expr_list = [e.strip() for e in str(row['expressions']).split(',')]
            new_taxonomy[row['new_category']][row['new_label']].extend(expr_list)
        
        # 중복 표현 제거
        for category, labels in new_taxonomy.items():
            for label, exprs in labels.items():
                new_taxonomy[category][label] = sorted(list(set(exprs)))
        
        return new_taxonomy, final_log_df
    else:
        return None, None


def main():
    """메인 실행 함수"""
    logging.info("===== 택소노미 재구성 시스템 시작 =====")

    # --- 경로 설정 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', 'data_processed', '0728_1900')
    
    input_path = os.path.join(data_dir, 'label_expression_mapping.csv')
    context_data_path = os.path.join(data_dir, 'labeled_queries_structured.csv')
    
    output_taxonomy_path = os.path.join(script_dir, 'taxonomy_v2.json')
    output_log_path = os.path.join(script_dir, 'reclassification_log.csv')

    # --- 데이터 로드 ---
    try:
        mapping_df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return
    
    context_map = build_context_map(context_data_path)
    if context_map is None:
        return

    # --- 재구성 실행 ---
    agent = get_llm_agent()
    new_taxonomy, log_df = restructure_taxonomy(mapping_df, agent, output_log_path, context_map)

    if new_taxonomy is not None and log_df is not None:
        # --- 최종 택소노미 파일 저장 ---
        try:
            with open(output_taxonomy_path, 'w', encoding='utf-8') as f:
                json.dump(new_taxonomy, f, ensure_ascii=False, indent=2)
            logging.info(f"성공적으로 새로운 택소노미 파일을 저장했습니다. 경로: {output_taxonomy_path}")
            
        except Exception as e:
            logging.error(f"택소노미 파일 저장 중 오류 발생: {e}")
            
        print("\n===== 재구성 작업 완료! =====")
        # 로그 파일은 이미 저장되었으므로 경로만 안내
        print(f"새로운 택소노미 파일: {output_taxonomy_path}")
        print(f"재분류 로그 파일: {output_log_path}")
        print("==========================")
    else:
        logging.info("작업이 완료되었거나 처리할 데이터가 없습니다.")

if __name__ == "__main__":
    main() 