import pandas as pd
import os
import json
import re
from collections import Counter
from tqdm import tqdm
from agent import TaxonomyRefinementAgent

class LabelingSystem:
    def __init__(self, input_file_path, output_dir_path):
        self.input_file_path = input_file_path
        self.output_dir = output_dir_path
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.df = self._load_data()
        self.taxonomy_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'taxonomy.json')
        self.taxonomy = self._load_taxonomy()
        self.agent = TaxonomyRefinementAgent()
        
        # tqdm 설정을 위한 pandas integration
        tqdm.pandas()

    def _load_data(self):
        try:
            df = pd.read_csv(self.input_file_path)
            # Ensure the query column is string type
            df.iloc[:, 0] = df.iloc[:, 0].astype(str)
            return df
        except FileNotFoundError:
            print(f"오류: 입력 파일 '{self.input_file_path}'을 찾을 수 없습니다.")
            exit()

    def _load_taxonomy(self):
        try:
            with open(self.taxonomy_file_path, 'r', encoding='utf-8') as f:
                print("저장된 taxonomy.json 파일을 불러옵니다.")
                return json.load(f)
        except FileNotFoundError:
            print("저장된 taxonomy.json 파일이 없습니다. 기본 택소노미로 시작합니다.")
            return self._get_initial_taxonomy()

    def _save_taxonomy(self):
        with open(self.taxonomy_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.taxonomy, f, indent=2, ensure_ascii=False)
        # print("택소노미가 파일에 저장되었습니다.") # 너무 자주 출력되므로 주석 처리

    def _get_initial_taxonomy(self):
        # In a real system, this could be loaded from a YAML or JSON file.
        return {
            '병원명': {'내이튼': ['내이튼', '네이튼', '이튼치과'], '윤민정원장': ['윤민정', '윤치과']},
            '지역': {'동탄': ['동탄'], '동탄역': ['동탄역'], '동탄2': ['2동탄', '동탄2'], '오산': ['오산'], '화성': ['화성']},
            '진료': {
                '임플란트': ['임플란트', '식립'], '신경치료': ['신경치료', '엔도', '근관치료'], 
                '재신경치료': ['재신경치료', '리엔도'], '충치': ['충치', '우식', '썩은', '썪은'],
                '레진': ['레진'], '크라운': ['크라운'], '인레이': ['인레이', '온레이'],
                '브릿지': ['브릿지'], '라미네이트': ['라미네이트'], '치아미백': ['미백', '실활치미백'],
                '발치': ['발치'], '뼈이식': ['뼈이식', '골이식'], '상악동거상술': ['상악동']
            },
            '증상': {
                '파손/깨짐': ['파절', '깨짐', '깨진', '금', '크랙', '금이 간'],
                '염증/고름': ['염증', '고름', '농', '물집'],
                '통증': ['통증', '아파', '아픈', '시림', '시큰'],
                '변색': ['변색', '까맣', '검게'],
                '흔들림': ['흔들', '흔들리는'],
                '퇴축': ['퇴축', '내려앉음']
            },
            '의도': {'비용문의': ['비용', '가격'], '추천요청': ['추천', '잘하는곳', '잘하는 곳'], '과정문의': ['과정', '기간', '순서'], '후기문의': ['후기']},
            '대상': {'소아': ['소아', '어린이', '아이']}
        }

    def _label_single_query(self, query):
        q_lower = str(query).replace('"', '').lower()
        matched_results = []
        
        flat_taxonomy = {}
        for category, labels in self.taxonomy.items():
            for label, keywords in labels.items():
                for keyword in keywords:
                    flat_taxonomy[keyword] = f"{category}:{label}"
        
        # 키워드를 길이 순으로 정렬하여 긴 키워드가 먼저 매칭되도록 함
        sorted_keywords = sorted(flat_taxonomy.keys(), key=len, reverse=True)

        # Multi-pass matching to handle overlapping keywords
        unlabeled_text = q_lower
        temp_unlabeled = unlabeled_text
        
        for keyword in sorted_keywords:
            if keyword in temp_unlabeled:
                label = flat_taxonomy[keyword]
                matched_results.append({'matched_expression': keyword, 'label': label})
                # Replace only the first occurrence to handle multiple instances
                temp_unlabeled = temp_unlabeled.replace(keyword, " " * len(keyword), 1)

        # Determine stage
        stage = '2단계:정보탐색'
        if any(res['label'].startswith('병원명:') for res in matched_results):
            stage = '4단계:병원인지'
        elif any(res['label'].startswith('지역:') for res in matched_results):
            stage = '3단계:지역탐색'
            
        final_unlabeled = re.sub(r'\s+', ' ', temp_unlabeled).strip()
        
        return json.dumps(matched_results, ensure_ascii=False), stage, final_unlabeled

    def run_single_labeling_pass(self):
        query_column = self.df.columns[0]
        
        # Use progress_apply for visual feedback
        results = self.df[query_column].progress_apply(self._label_single_query)
        
        self.df[['labels(json)', 'stage', 'unlabeled_expressions']] = pd.DataFrame(results.tolist(), index=self.df.index)

    def get_unlabeled_report(self):
        unlabeled_texts = self.df['unlabeled_expressions'].dropna().tolist()
        all_unlabeled_words = []
        for text in unlabeled_texts:
            # Split by whitespace and filter out short/numeric tokens
            words = [word for word in text.split() if len(word) > 1 and not word.isnumeric()]
            all_unlabeled_words.extend(words)

        if not all_unlabeled_words:
            return pd.DataFrame(columns=['Expression', 'Count'])

        unlabeled_counts = Counter(all_unlabeled_words)
        unlabeled_df = pd.DataFrame(unlabeled_counts.most_common(), columns=['Expression', 'Count'])
        return unlabeled_df

    def refine_taxonomy(self, expressions_df):
        update_count = 0
        existing_labels = list(self.taxonomy.keys())
        
        print("LLM 에이전트를 통해 택소노미 개선을 시작합니다...")
        for _, row in tqdm(expressions_df.iterrows(), total=expressions_df.shape[0], desc="Analyzing Expressions"):
            expression = row['Expression']
            decision = self.agent.decide(expression, self.taxonomy)
            
            if decision['action'] == 'MAP_TO_EXISTING':
                target_label_str = decision.get('target_label')
                # 방어 코드: target_label 형식이 올바른지 확인
                if target_label_str and ':' in target_label_str:
                    cat, label = target_label_str.split(':', 1)
                    if cat in self.taxonomy and label in self.taxonomy[cat]:
                        if expression not in self.taxonomy[cat][label]:
                            self.taxonomy[cat][label].append(expression)
                            update_count += 1
                    else:
                        print(f"경고: 에이전트가 지정한 target_label '{target_label_str}'이(가) 택소노미에 존재하지 않습니다. 생성을 시도합니다.")
                        # CREATE_NEW 로직과 유사하게 처리
                        self.taxonomy.setdefault(cat, {})
                        self.taxonomy[cat].setdefault(label, [])
                        if expression not in self.taxonomy[cat][label]:
                            self.taxonomy[cat][label].append(expression)
                            update_count += 1
                else:
                    print(f"경고: 에이전트가 생성한 target_label의 형식이 올바르지 않습니다 ('Category:Label' 필요): '{target_label_str}'. 처리를 건너뜁니다.")

            elif decision['action'] == 'CREATE_NEW':
                new_label_str = decision.get('new_label')
                if new_label_str and ':' in new_label_str:
                    cat, label = new_label_str.split(':', 1)
                    if cat not in self.taxonomy:
                        self.taxonomy[cat] = {}
                    if label not in self.taxonomy[cat]:
                        self.taxonomy[cat][label] = []
                    if expression not in self.taxonomy[cat][label]:
                        self.taxonomy[cat][label].append(expression)
                        update_count += 1
                else:
                    print(f"경고: 에이전트가 생성한 new_label의 형식이 올바르지 않습니다 ('Category:Label' 필요): '{new_label_str}'. 처리를 건너뜁니다.")
        
        if update_count > 0:
            self._save_taxonomy() # 택소노미에 변경이 있을 때만 저장
            
        return update_count

    def generate_final_reports(self):
        # 최종 택소노미 저장 (리포트 생성 시점에도 한번 더 저장)
        self._save_taxonomy()

        # Report 1: Detailed Labeled Queries
        report1_path = os.path.join(self.output_dir, 'labeled_queries_structured.csv')
        report1_df = self.df.rename(columns={self.df.columns[0]: 'searchQuery'})
        report1_df[['searchQuery', 'labels(json)', 'stage']].to_csv(report1_path, index=False, encoding='utf-8-sig')

        # Report 2: Label-Expression Mapping
        report2_path = os.path.join(self.output_dir, 'label_expression_mapping.csv')
        mapping_list = []
        for category, labels in self.taxonomy.items():
            for label, expressions in labels.items():
                mapping_list.append({
                    'Category': category,
                    'Label': label,
                    'Matched Expressions': ', '.join(sorted(list(set(expressions))))
                })
        mapping_df = pd.DataFrame(mapping_list).sort_values(by=['Category', 'Label'])
        mapping_df.to_csv(report2_path, index=False, encoding='utf-8-sig')

        # Report 3: Unlabeled Expressions Report
        report3_path = os.path.join(self.output_dir, 'unlabeled_expressions_report.csv')
        unlabeled_df = self.get_unlabeled_report()
        unlabeled_df.to_csv(report3_path, index=False, encoding='utf-8-sig')
        
        # Report 4: System Summary
        summary = f"""
        --- 시스템 실행 결과 요약 ---
        - 총 처리된 검색어: {len(self.df)}개
        - 최종 택소노미 카테고리 수: {len(self.taxonomy)}개
        - 최종 택소노미 라벨 수: {sum(len(labels) for labels in self.taxonomy.values())}개
        - 최종 미분류 표현 (고유): {len(unlabeled_df)}개

        --- 생성된 파일 ---
        1. 상세 라벨링 파일: {report1_path}
        2. 라벨-표현 매핑: {report2_path}
        3. 미분류 표현 리포트: {report3_path}
        """
        report4_path = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report4_path, 'w', encoding='utf-8') as f:
            f.write(summary)
            
        return summary 