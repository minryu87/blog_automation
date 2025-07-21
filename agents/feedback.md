### Human Feedback for the SEO Agent Team

**Instructions for the user:**
<!-- 
기존 지시사항들:
"score_로 시작하는 기존 점수 컬럼들은 사용하지 말고, 반드시 'post_body' 컬럼의 원본 텍스트를 직접 분석하여 주제 관련성을 파악하는 'semantic_analysis'를 수행하라."
"post_body는 사용하지 말고, category_keywords,morpheme_words 데이터를 활용하여 주제 관련성을 파악하는 'semantic_analysis'를 수행하라. "
-->

이번 목표는 '성과가 좋은 포스트'와 '성과가 낮은 포스트' 간의 의미론적(semantic) 차이를 발견하여, 성과를 예측하는 새로운 피처를 발굴하는 것입니다.

다음 단계를 따라 작업을 진행하세요:

1.  **데이터셋 정의**: `blog_automation/data/data_processed/master_post_data.csv` 파일을 입력으로 사용합니다.

2.  **그룹 정의**:
    *   **고성과 그룹(High-performing Group)**:
        *   우리 포스트(`source` == 'ours') 중 `non_brand_inflow` 기준 상위 30% 포스트.
        *   모든 경쟁사 포스트 (`source` == 'competitor').
    *   **저성과 그룹(Low-performing Group)**:
        *   우리 포스트(`source` == 'ours') 중 `non_brand_inflow` 기준 하위 30% 포스트.

3.  **의미론적 분석 가설 수립**:
    *   '고성과 그룹'의 `post_body` 및 `morpheme_words` 데이터에서 공통적으로 발견되지만, '저성과 그룹'에서는 누락된 의미론적 패턴(예: 특정 키워드 조합, 주제의 흐름, 문장 구조 등)을 찾아내는 가설을 수립하세요.

4.  **피처 생성 및 검증**:
    *   위 가설을 검증할 수 있는 새로운 피처를 생성하는 Python 코드를 작성하세요.
    *   생성된 피처와 우리 포스트의 성과 지표(`non_brand_inflow`, `non_brand_average_ctr`) 간의 상관 관계를 분석하여 가설의 유효성을 최종 평가하세요.
