import os
import json
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
import signal
import sys

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 우아한 종료를 위한 글로벌 플래그 및 핸들러 ---
keep_running = True
def signal_handler(sig, frame):
    """Ctrl+C 입력을 감지하여 안전한 종료를 준비합니다."""
    global keep_running
    if keep_running:
        logging.warning("\n[Ctrl+C 감지] 현재 처리를 완료한 후 저장을 시도합니다. 즉시 종료하려면 다시 한번 눌러주세요.")
        keep_running = False
    else:
        logging.warning("\n[강제 종료] 즉시 프로그램을 종료합니다.")
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

# --- 상수 정의 ---
# 1. 분석의 기준이 될 핵심 '관심진료' 리스트 (사용자와 합의된 최종 버전)
CORE_INTEREST_AREAS = [
    '치아보존술', # 전략적 중요도 최상위
    '임플란트',
    '충치치료',   # '일반진료'에서 세분화
    '신경치료',   # '일반진료'에서 세분화
    '잇몸치료',   # '일반진료'에서 세분화
    '심미치료'
]

# 2. 통계 분석을 위한 PMI 임계값
PMI_THRESHOLD = 0.1 # 0보다 큰 값으로 설정하여 우연 이상의 연관성만 선택

# --- LLM 에이전트 설정 ---
def get_llm_agent():
    """Gemini LLM 에이전트를 초기화하고 반환합니다."""
    try:
        load_dotenv(find_dotenv())
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        llm = Gemini(
            id=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
            api_key=os.getenv("GEMINI_API_KEY")
        )
        return Agent(model=llm)
    except Exception as e:
        logging.error(f"LLM 에이전트 초기화 실패: {e}")
        return None

# --- 데이터 로딩 및 전처리 ---
def load_and_preprocess_data(labeled_queries_path):
    """
    라벨링된 쿼리 데이터를 로드하고, 각 쿼리별로 라벨 리스트를 추출합니다.
    """
    try:
        df = pd.read_csv(labeled_queries_path)
        df.dropna(subset=['labels(json)'], inplace=True)

        all_query_labels = []
        for _, row in df.iterrows():
            try:
                # labels(json) 컬럼에서 'label' 값만 추출
                labels_json = json.loads(row['labels(json)'])
                # 'Category:Label' 형태의 전체 라벨명 추출
                labels = [item['label'] for item in labels_json]
                if labels:
                    all_query_labels.append(labels)
            except (json.JSONDecodeError, TypeError):
                continue
        
        logging.info(f"{len(all_query_labels)}개의 유효한 쿼리에서 라벨 리스트를 추출했습니다.")
        return all_query_labels
    except FileNotFoundError:
        logging.error(f"입력 파일을 찾을 수 없습니다: {labeled_queries_path}")
        return None

# --- 1단계: 통계 기반 연관성 분석 ---
def generate_associations_statistically(all_query_labels, all_unique_labels):
    """
    라벨간 동시 등장 빈도와 PMI 점수를 기반으로 연관 관계 초안을 생성합니다.
    """
    logging.info("1단계: 통계 기반 연관성 분석 시작...")
    
    label_freq = defaultdict(int)
    co_occurrence_freq = defaultdict(int)
    num_queries = len(all_query_labels)

    # 1. 빈도 계산
    for labels in tqdm(all_query_labels, desc="빈도 계산 중"):
        for label in set(labels):
            label_freq[label] += 1
        for l1, l2 in combinations(set(labels), 2):
            co_occurrence_freq[(l1, l2)] += 1
            co_occurrence_freq[(l2, l1)] += 1
            
    # 2. PMI 점수 계산 및 연관성 매핑
    associations = defaultdict(list)
    associated_labels = set()
    
    # 관심 진료 영역에 속하지 않는 라벨만 대상으로 함
    other_labels = [l for l in all_unique_labels if l not in CORE_INTEREST_AREAS]

    for label in tqdm(other_labels, desc="PMI 계산 및 연관성 매핑 중"):
        if not keep_running:
            logging.info("종료 신호를 감지하여 통계 분석을 중단합니다.")
            break
            
        pmi_scores = {}
        for area in CORE_INTEREST_AREAS:
            # 두 라벨의 동시 등장 확률
            p_xy = co_occurrence_freq.get((label, area), 0) / num_queries
            if p_xy == 0:
                continue

            # 각 라벨의 등장 확률
            p_x = label_freq.get(label, 0) / num_queries
            p_y = label_freq.get(area, 0) / num_queries
            
            if p_x * p_y == 0:
                continue
                
            pmi = np.log2(p_xy / (p_x * p_y))
            pmi_scores[area] = pmi
        
        if pmi_scores:
            # 가장 높은 PMI 점수를 가진 관심 진료 영역을 찾음
            best_area = max(pmi_scores, key=pmi_scores.get)
            if pmi_scores[best_area] > PMI_THRESHOLD:
                associations[best_area].append(label)
                associated_labels.add(label)

    logging.info(f"통계 분석을 통해 {len(associated_labels)}개의 라벨을 관심 진료와 연결했습니다.")
    return associations, associated_labels

# --- 2단계: LLM 기반 보강 ---
def augment_associations_with_llm(associations, associated_labels, all_unique_labels, agent):
    """
    통계적으로 연결되지 않은 라벨들을 LLM을 이용해 분류하고 연관 관계를 보강합니다.
    """
    unassociated_labels = [l for l in all_unique_labels if l not in associated_labels and l not in CORE_INTEREST_AREAS]
    logging.info(f"2단계: LLM 기반 연관성 보강 시작 (대상 라벨: {len(unassociated_labels)}개)")

    if not agent:
        logging.warning("LLM 에이전트가 없어 2단계를 건너뜁니다.")
        return associations

    prompt_template = """
    당신은 수십 년 경력의 치과 보존과 전문의입니다.
    주어진 '치과 용어'가 아래의 '핵심 진료 분야' 중 어떤 것과 가장 밀접하게 관련되어 있는지 판단해야 합니다.
    오직 하나의 가장 관련 깊은 분야만 선택해야 합니다.

    '핵심 진료 분야' 리스트: {interest_areas}

    '치과 용어': "{label}"

    가장 관련 깊은 '핵심 진료 분야'의 이름 하나만 정확히 답변해주세요. (예: 임플란트)
    만약 어떤 분야와도 명확한 관련이 없다면 '없음'이라고 답변해주세요.
    """

    llm_associated_count = 0
    for label in tqdm(unassociated_labels, desc="LLM으로 미분류 라벨 처리 중"):
        if not keep_running:
            logging.info("종료 신호를 감지하여 LLM 보강 작업을 중단합니다.")
            break

        try:
            prompt = prompt_template.format(
                interest_areas=', '.join(CORE_INTEREST_AREAS),
                label=label
            )
            response = agent.run(prompt)
            best_area = response.content.strip()

            if best_area in CORE_INTEREST_AREAS:
                associations[best_area].append(label)
                llm_associated_count += 1
        except Exception as e:
            logging.error(f"'{label}' 처리 중 LLM 오류 발생: {e}")
            continue
    
    logging.info(f"LLM을 통해 추가로 {llm_associated_count}개의 라벨을 연결했습니다.")
    return associations


def main():
    """메인 실행 함수"""
    logging.info("===== 'label_associations.json' 자동 생성 시스템 시작 =====")

    # --- 경로 설정 ---
    script_dir = os.path.dirname(os.path.abspath(__file__)) # /searchquery_analysis
    
    # 입력 데이터 경로 수정
    # 통계 분석을 위해 v1 라벨링 쿼리 데이터를 계속 사용
    labeled_queries_path = os.path.join(script_dir, 'data', 'data_processed', '0728_1900', 'labeled_queries_structured.csv')
    # 재구성된 v2 택소노미를 기준으로 전체 라벨 목록을 가져옴
    v2_taxonomy_path = os.path.join(script_dir, 'taxonomy_v2.json')
    
    output_path = os.path.join(script_dir, 'label_associations.json')

    # --- 데이터 로드 ---
    # 1. 통계 분석을 위한 컨텍스트 로드 (v1 데이터 기반)
    all_query_labels = load_and_preprocess_data(labeled_queries_path)
    if not all_query_labels:
        return
        
    # 2. 전체 고유 라벨 목록 로드 (v2 택소노미 기반)
    try:
        with open(v2_taxonomy_path, 'r', encoding='utf-8') as f:
            taxonomy_v2 = json.load(f)
        
        all_unique_labels = []
        for category, labels in taxonomy_v2.items():
            for label in labels.keys():
                all_unique_labels.append(f"{category}:{label}")
        logging.info(f"'{os.path.basename(v2_taxonomy_path)}'에서 {len(all_unique_labels)}개의 고유 라벨(v2)을 로드했습니다.")
    except FileNotFoundError:
        logging.error(f"재구성된 택소노미 파일을 찾을 수 없습니다: {v2_taxonomy_path}")
        return

    # --- 1단계 실행 ---
    associations, associated_labels = generate_associations_statistically(all_query_labels, all_unique_labels)
    
    # --- 2단계 실행 ---
    agent = get_llm_agent()
    final_associations = augment_associations_with_llm(associations, associated_labels, all_unique_labels, agent)

    # --- 결과 저장 ---
    # 가독성을 위해 각 리스트를 정렬
    for area in final_associations:
        final_associations[area] = sorted(list(set(final_associations[area])))

    # 최종 결과물 저장
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_associations, f, ensure_ascii=False, indent=2)
        logging.info(f"성공적으로 'label_associations.json' 파일을 생성했습니다. 경로: {output_path}")
    except Exception as e:
        logging.error(f"결과 파일 저장 중 오류 발생: {e}")

    # --- 요약 출력 ---
    print("\n===== 최종 요약 =====")
    for area, related_labels in final_associations.items():
        print(f"\n- {area} ({len(related_labels)}개):")
        # 10개까지만 예시로 출력
        print(f"  {related_labels[:10]}")
    print("\n=====================")


if __name__ == "__main__":
    main() 