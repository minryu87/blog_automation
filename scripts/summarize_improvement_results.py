import pandas as pd
import json
import os
import glob

def calculate_expected_inflow(score):
    """주어진 모델 점수에 대해 기대 검색 유입률을 계산합니다."""
    if score is None:
        return 0.0
    # y = -77.04x² + 13.61x + 0.07
    return (-77.04 * (score**2)) + (13.61 * score) + 0.07

def format_features_md(features_dict):
    """피처 딕셔너리를 마크다운 리스트로 예쁘게 변환합니다."""
    if not isinstance(features_dict, dict):
        return ""
    return "\n".join([f"    * `{key}`: **{value:.4f}**" for key, value in features_dict.items()])

def summarize_results_to_markdown(file_paths, prediction_csv_path):
    """
    여러 개의 edit_history.json 파일을 분석하여 결과를 Markdown 보고서로 저장합니다.
    """
    # Load prediction data and calculate initial ranks
    try:
        prediction_df = pd.read_csv(prediction_csv_path)
        prediction_df['post_id'] = prediction_df['post_id'].astype(str) # Join을 위해 str으로 변환
        prediction_df['initial_rank'] = prediction_df['predicted_ctr'].rank(method='dense', ascending=False).astype(int)
        prediction_df['initial_percentile_top'] = (prediction_df['initial_rank'] / len(prediction_df)) * 100
        total_posts = len(prediction_df)
    except FileNotFoundError:
        print(f"오류: 순위 계산을 위한 예측 파일({prediction_csv_path})을 찾을 수 없습니다.")
        return

    report_content = "# 포스트 개선 결과 요약 보고서\n\n"
    report_content += "AI 편집 에이전트에 의해 자동 개선된 포스트들의 초기 상태와 최고 성과 달성 상태를 비교 분석한 결과입니다.\n"

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"파일을 읽는 중 오류 발생: {file_path} - {e}")
            continue

        post_id = str(data.get('post_id')) # JSON에서 읽은 ID도 str으로 통일
        initial_data = data.get('initial', {})

        # 초기 순위 정보 가져오기
        rank_info = prediction_df[prediction_df['post_id'] == post_id]
        if rank_info.empty:
            print(f"경고: 예측 파일에서 post_id {post_id}를 찾을 수 없어 순위를 계산할 수 없습니다.")
            initial_rank, initial_percentile_top = 'N/A', 'N/A'
        else:
            initial_rank = rank_info['initial_rank'].iloc[0]
            initial_percentile_top = rank_info['initial_percentile_top'].iloc[0]

        # 최고 성과 버전 찾기
        best_version_key = 'initial'
        best_score = initial_data.get('model_score', float('-inf')) if initial_data else float('-inf')

        for key, value in data.items():
            if key.startswith('v') and isinstance(value, dict):
                current_score = value.get('model_score', float('-inf'))
                if current_score > best_score:
                    best_score = current_score
                    best_version_key = key
        
        best_version_data = data.get(best_version_key, {})

        # 개선 후 예상 순위 계산
        improved_rank = (prediction_df['predicted_ctr'] > best_score).sum() + 1
        improved_percentile_top = (improved_rank / total_posts) * 100

        # 기대 유입률 계산
        initial_inflow_rate = calculate_expected_inflow(initial_data.get('model_score'))
        best_inflow_rate = calculate_expected_inflow(best_version_data.get('model_score'))
        
        # 유입률 상승률 계산
        inflow_improvement_rate = 0.0
        if initial_inflow_rate > 0:
            inflow_improvement_rate = ((best_inflow_rate / initial_inflow_rate) - 1) * 100
        elif best_inflow_rate > 0:
            inflow_improvement_rate = float('inf') # 초기값이 0일 경우 무한대로 표시

        # 보고서 섹션 추가
        report_content += f"\n---\n\n"
        report_content += f"## 🚀 포스트 ID: {post_id}\n\n"
        
        # 1. 최초 상태
        report_content += "### 1. 최초 상태\n"
        report_content += f"*   **전체 순위**: **{initial_rank}위 / {total_posts}개** (상위 {initial_percentile_top:.1f}%)\n"
        report_content += f"*   **모델 점수**: `{initial_data.get('model_score', 0):.4f}`\n"
        report_content += f"*   **기대 검색 유입률**: **{initial_inflow_rate:.2%}**\n"
        report_content += f"*   **제목**: {initial_data.get('post_title', 'N/A')}\n"
        report_content += f"*   **주요 피처 값**:\n{format_features_md(initial_data.get('feature_values'))}\n"
        report_content += f"*   **본문**:\n```\n{initial_data.get('post_body', '')}\n```\n\n"

        # 2. 최고 성과 달성
        report_content += f"### 2. 최고 성과 달성 ({best_version_key})\n"
        report_content += f"*   **개선 후 예상 순위**: **{improved_rank}위 / {total_posts}개** (상위 {improved_percentile_top:.1f}%)\n"
        if isinstance(initial_rank, (int, float)) and not pd.isna(initial_rank):
             report_content += f"*   **순위 상승**: **{int(initial_rank) - improved_rank}** 계단 상승 📈\n"
        report_content += f"*   **모델 점수**: `{best_version_data.get('model_score', 0):.4f}`\n"
        report_content += f"*   **기대 검색 유입률**: **{best_inflow_rate:.2%}**\n"
        if inflow_improvement_rate == float('inf'):
            report_content += f"*   **기대 검색 유입률 상승률**: **+∞%** (초기 유입률 0에서 증가) 🚀\n"
        else:
            report_content += f"*   **기대 검색 유입률 상승률**: **+{inflow_improvement_rate:.1f}%** 🚀\n"
        report_content += f"*   **제목**: {best_version_data.get('post_title', 'N/A')}\n"
        report_content += f"*   **주요 피처 값**:\n{format_features_md(best_version_data.get('feature_values'))}\n"
        report_content += f"*   **본문**:\n```\n{best_version_data.get('post_body', '')}\n```\n"

    output_filename = 'post_improvement_summary.md'
    output_path = os.path.join(os.path.dirname(file_paths[0]), output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"마크다운 보고서가 성공적으로 저장되었습니다: {output_path}")

if __name__ == '__main__':
    prediction_file = 'blog_automation/data/post_edit/20250722_143801_feature_with_prediction.csv'
    # 제공된 파일 목록을 직접 사용
    json_files = [
        'blog_automation/data/post_edit/20250722_153854/223155543711_edit_history.json',
        'blog_automation/data/post_edit/20250722_153854/223235174542_edit_history.json',
        'blog_automation/data/post_edit/20250722_153854/223335804761_edit_history.json',
        'blog_automation/data/post_edit/20250722_153854/223580589189_edit_history.json',
        'blog_automation/data/post_edit/20250722_153854/223620072636_edit_history.json'
    ]
    summarize_results_to_markdown(json_files, prediction_file) 