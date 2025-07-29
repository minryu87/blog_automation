import pandas as pd
import os
import json
from collections import Counter

def analyze_structured_labels():
    """
    구조적으로 라벨링된 검색어 데이터를 분석하여
    1. 라벨-표현 매핑 파일 생성
    2. 라벨 미부여 표현 리포트 파일 생성
    """
    # 경로 설정
    workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(workspace_path, 'data', 'data_processed', 'labeled_queries_structured.csv')
    output_dir = os.path.join(workspace_path, 'data', 'data_processed')
    mapping_output_file = os.path.join(output_dir, 'label_expression_mapping.csv')
    unlabeled_output_file = os.path.join(output_dir, 'unlabeled_expressions_report.csv')

    # 데이터 로드
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"오류: 분석할 파일 '{input_file}'을 찾을 수 없습니다.")
        print("먼저 label_search_queries.py를 실행하여 구조화된 라벨링 파일을 생성해주세요.")
        return

    print("라벨-표현 매핑 및 미부여 표현 분석을 시작합니다...")

    # --- 1. 라벨-표현 매핑 분석 ---
    label_to_expressions = {}
    
    # NaN 값을 처리하며 JSON 파싱
    valid_labels = df['labels(json)'].dropna()
    for json_str in valid_labels:
        try:
            labels_list = json.loads(json_str)
            for item in labels_list:
                label = item['label']
                expression = item['matched_expression']
                if label not in label_to_expressions:
                    label_to_expressions[label] = set()
                label_to_expressions[label].add(expression)
        except json.JSONDecodeError:
            # 파싱 오류가 있는 경우 건너뛰기
            continue

    mapping_list = []
    for label, expressions in label_to_expressions.items():
        category, label_name = label.split(':', 1)
        mapping_list.append({
            'Category': category,
            'Label': label_name,
            'Matched Expressions': ', '.join(sorted(list(expressions)))
        })
    
    mapping_df = pd.DataFrame(mapping_list)
    mapping_df = mapping_df.sort_values(by=['Category', 'Label']).reset_index(drop=True)
    mapping_df.to_csv(mapping_output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n라벨-표현 매핑 파일이 다음 경로에 저장되었습니다: {mapping_output_file}")
    print("\n--- 라벨-표현 매핑 결과 미리보기 ---")
    print(mapping_df.head())

    # --- 2. 라벨 미부여 표현 분석 ---
    unlabeled_texts = df['unlabeled_expressions'].dropna().tolist()
    all_unlabeled_words = []
    for text in unlabeled_texts:
        all_unlabeled_words.extend(text.split())

    unlabeled_counts = Counter(all_unlabeled_words)
    unlabeled_df = pd.DataFrame(unlabeled_counts.most_common(), columns=['Expression', 'Count'])
    unlabeled_df.to_csv(unlabeled_output_file, index=False, encoding='utf-8-sig')

    print(f"\n라벨 미부여 표현 리포트가 다음 경로에 저장되었습니다: {unlabeled_output_file}")
    print("\n--- 라벨 미부여 표현 빈도수 상위 10개 ---")
    print(unlabeled_df.head(10))

if __name__ == "__main__":
    analyze_structured_labels() 