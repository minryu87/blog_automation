import pandas as pd
import os
import json
from collections import Counter
import re

def get_taxonomy():
    """
    분류 체계(택소노미)를 정의하는 함수.
    """
    taxonomy = {
        'hospital_name': {
            '내이튼': ['내이튼', '네이튼', 'naeiteun', '이튼치과'],
            '윤민정원장': ['윤민정', '윤치과'],
            '기타병원': ['포인트치과', '정이든치과', '더새로운치과', '나무치과', '이즈치과']
        },
        'location': {
            '동탄역': ['동탄역'],
            '동탄2': ['동탄2', '2동탄'],
            '동탄': ['동탄'],
            '오산': ['오산'],
            '화성': ['화성'],
            '랜드마크': ['라스플로레스']
        },
        'treatment': {
            '임플란트': {
                'root': ['임플란트', '임프란트', '인플란트'],
                'children': {
                    '픽스쳐': ['픽스쳐', '픽스처', 'fixture'],
                    '뼈이식': ['뼈이식', '골이식'],
                    '상악동거상술': ['상악동', 'sinus'],
                    '재수술': ['재수술', '재식립', '부작용']
                }
            },
            '신경치료': {
                'root': ['신경치료', '근관치료', '엔도', 'endo'],
                'children': {
                    '재신경치료': ['재신경치료', '리엔도', 're endo'],
                    '치근단절제술': ['치근단절제술', 'apicoectomy'],
                    '치수복조': ['치수복조', 'pulpotomy'],
                    'MB2': ['mb2']
                }
            },
            '보철치료': {
                'root': ['보철'],
                'children': {
                    '크라운': ['크라운', 'crown'],
                    '브릿지': ['브릿지', '브리지'],
                    '인레이': ['인레이', '온레이', 'inlay', 'onlay'],
                    '틀니': ['틀니', '오버덴쳐', 'denture']
                }
            },
            '심미치료': {
                'root': ['심미'],
                'children': {
                    '라미네이트': ['라미네이트', 'laminate'],
                    '치아미백': ['미백', '실활치미백'],
                    '잇몸성형': ['잇몸성형', '잇몸미백'],
                    '레진': ['레진', '다이아스테마', '테세라'],
                }
            },
            '보존치료': {
                'root': [],
                'children': {
                    '충치': ['충치', '우식', '썩은', '썪은'],
                    '실란트': ['실란트', 'sealant'],
                }
            },
            '치주치료': {
                'root': ['치주'],
                'children': {
                    '스케일링': ['스케일링', '치석제거'],
                    '잇몸치료': ['잇몸치료', '치주염', '치은염'],
                }
            },
            '기타진료': {
                'root': [],
                'children': {
                    '발치': ['발치'],
                    '사랑니': ['사랑니'],
                    '교정': ['교정'],
                    '검진': ['검진', '정기검진', '구강검진']
                }
            }
        },
        'symptom': {
            '파손/깨짐': ['파절', '깨진', '깨졌', '부러진', '금간', '크랙', '금이 간'],
            '염증/고름': ['염증', '고름', '농', '물집', '병소'],
            '통증': ['통증', '아파', '아픈', '욱신', '시림', '시큰'],
            '변색': ['변색', '까맣', '검게', '누런'],
            '흔들림': ['흔들', '흔드'],
            '붓기': ['붓기', '부은', '부었', '부어'],
            '이물질': ['이물질', '음식물'],
            '퇴축': ['퇴축', '내려앉음']
        },
        'anatomy': {
            '앞니': ['앞니', '전치'],
            '어금니': ['어금니', '구치', '소구치'],
            '송곳니': ['송곳니'],
            '잇몸': ['잇몸', '치은', '치조골']
        },
        'intent': {
            '비용문의': ['비용', '가격', '얼마'],
            '추천요청': ['추천', '잘하는곳', '잘하는 곳', '잘하는치과', '잘하는 치과'],
            '과정문의': ['과정', '순서', '기간'],
            '후기문의': ['후기', '리뷰']
        },
        'target_patient': {
            '소아': ['소아', '어린이', '아이', '애기'],
            '성인': ['성인'],
            '임산부': ['임산부']
        },
        'quality': {
            'URL': ['http', 'www', '.com', '.net', '.kr'],
            '오타': ['ㅊ콰', '치거ㅏ', '치괴', '부근치과']
        }
    }
    return taxonomy

def label_query(query, taxonomy):
    # 원본 쿼리에서 "" 제거하고 소문자로 변환
    q_lower = query.replace('"', '').lower()
    
    matched_results = []
    unlabeled_text = q_lower

    # 모든 택소노미 키워드를 순회하며 라벨링
    all_categories = {
        '병원명': taxonomy['hospital_name'], '지역': taxonomy['location'], 
        '증상': taxonomy['symptom'], '위치': taxonomy['anatomy'], '의도': taxonomy['intent'],
        '대상': taxonomy['target_patient'], '품질': taxonomy['quality']
    }

    # 일반 카테고리
    for cat_name, category in all_categories.items():
        for label, keywords in category.items():
            for keyword in keywords:
                if keyword in q_lower:
                    matched_results.append({'matched_expression': keyword, 'label': f"{cat_name}:{label}"})
                    unlabeled_text = unlabeled_text.replace(keyword, '')

    # 계층적 진료 카테고리
    for main_cat, details in taxonomy['treatment'].items():
        # 하위 카테고리
        for sub_label, keywords in details.get('children', {}).items():
            for keyword in keywords:
                if keyword in q_lower:
                    matched_results.append({'matched_expression': keyword, 'label': f"진료:{main_cat}>{sub_label}"})
                    unlabeled_text = unlabeled_text.replace(keyword, '')
        # 상위 카테고리
        for keyword in details.get('root', []):
            if keyword in q_lower:
                matched_results.append({'matched_expression': keyword, 'label': f"진료:{main_cat}"})
                unlabeled_text = unlabeled_text.replace(keyword, '')

    # 중복 제거 (가장 긴 매칭을 우선으로)
    # 예를 들어 '동탄역'과 '동탄'이 모두 매칭된 경우 '동탄역'만 남기도록 함
    unique_matches = []
    # matched_expression을 기준으로 정렬 (긴 것이 앞으로)
    matched_results.sort(key=lambda x: len(x['matched_expression']), reverse=True)
    seen_expressions = set()
    for res in matched_results:
        if res['matched_expression'] not in seen_expressions:
            # 더 긴 표현이 이미 처리되었는지 확인
            is_substring = False
            for seen in seen_expressions:
                if res['matched_expression'] in seen:
                    is_substring = True
                    break
            if not is_substring:
                unique_matches.append(res)
                seen_expressions.add(res['matched_expression'])

    # 고객 여정 단계 결정
    stage = '2단계:정보탐색'
    if any(res['label'].startswith('병원명:') for res in unique_matches):
        stage = '4단계:병원인지'
    elif any(res['label'].startswith('지역:') for res in unique_matches):
        stage = '3단계:지역탐색'

    # 라벨 미부여 표현 정리
    # 공백, 특수문자, 조사 등 제거
    unlabeled_expressions = re.sub(r'[\s\~\!\@\#\$\%\^\&\*\(\)\-\_\+\=\[\]\{\}\;\:\'\"\,\.\<\>\/\?]+', ' ', unlabeled_text).strip()
    # 추가적으로 제거할 불용어
    stopwords = ['사이', '경우', '문의', '대해', '관련']
    for stopword in stopwords:
        unlabeled_expressions = unlabeled_expressions.replace(stopword, '').strip()
    
    unlabeled_expressions = ' '.join(unlabeled_expressions.split())

    return unique_matches, stage, unlabeled_expressions


def main():
    # 경로 설정
    workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(workspace_path, 'data', 'data_input', 'searchQuery.csv')
    output_dir = os.path.join(workspace_path, 'data', 'data_processed')
    output_file = os.path.join(output_dir, 'labeled_queries_structured.csv')

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_file}'을 찾을 수 없습니다.")
        return

    query_column = df.columns[0]
    print(f"'{query_column}' 컬럼을 사용하여 라벨링을 시작합니다...")
    
    taxonomy = get_taxonomy()
    
    results = df[query_column].astype(str).apply(lambda q: label_query(q, taxonomy))
    
    # DataFrame에 새로운 컬럼 추가
    df['labels(json)'] = [json.dumps(res[0], ensure_ascii=False) for res in results]
    df['stage'] = [res[1] for res in results]
    df['unlabeled_expressions'] = [res[2] for res in results]
    
    # 최종 결과는 요청된 컬럼만 포함
    final_df = df[[query_column, 'labels(json)', 'stage', 'unlabeled_expressions']]
    final_df = final_df.rename(columns={query_column: 'searchQuery'})
    
    # 결과 저장
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"상세 라벨링 작업이 완료되었습니다. 결과가 다음 파일에 저장되었습니다: {output_file}")
    print("\n--- 결과 미리보기 (labeled_queries_structured.csv) ---")
    print(final_df.head())


if __name__ == "__main__":
    main() 