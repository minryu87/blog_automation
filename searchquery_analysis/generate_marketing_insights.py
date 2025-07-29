import pandas as pd
import os
import json
import ast
import logging
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- 로깅 및 시각화 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_matplotlib_korean_font():
    """Matplotlib에서 한글 폰트를 설정합니다."""
    font_files = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    korean_font_path = None
    for font_file in font_files:
        if 'AppleSDGothicNeo' in font_file:
            korean_font_path = font_file
            break
        elif 'NanumGothic' in font_file:
            korean_font_path = font_file
            break
        
    if korean_font_path:
        font_manager.fontManager.addfont(korean_font_path)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=korean_font_path).get_name()
        logging.info(f"한글 폰트가 설정되었습니다: {korean_font_path}")
    else:
        logging.warning("사용 가능한 한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
    
    plt.rcParams['axes.unicode_minus'] = False

setup_matplotlib_korean_font()


def load_and_merge_data(base_path):
    """네 개의 핵심 데이터 소스를 로드하고 통합하여 마스터 데이터프레임을 생성합니다."""
    logging.info("데이터 로드 및 통합을 시작합니다.")
    try:
        labeled_queries_path = os.path.join(base_path, 'data/data_processed/0728_2300/final_labeled_queries.csv')
        search_console_path = os.path.join(base_path, 'data/search_console_analysis.csv')
        post_query_path = os.path.join(os.path.dirname(base_path), 'data/data_input/post-searchQuery.csv')
        post_meta_path = os.path.join(os.path.dirname(base_path), 'data/data_processed/agent_base_dataset.csv')

        labeled_df = pd.read_csv(labeled_queries_path)
        performance_df = pd.read_csv(search_console_path)
        post_query_df = pd.read_csv(post_query_path)
        post_meta_df = pd.read_csv(post_meta_path)
        logging.info("모든 CSV 파일 로드 완료.")
    except FileNotFoundError as e:
        logging.error(f"필수 파일을 찾을 수 없습니다: {e}")
        return None, None, 0

    performance_df.rename(columns={'검색어': 'searchQuery'}, inplace=True)
    performance_df['date'] = pd.to_datetime(performance_df['월'], format='%Y-%m')
    num_months = performance_df['date'].dt.to_period('M').nunique()
    
    performance_agg_df = performance_df.groupby('searchQuery').agg(
        total_search_volume=('네이버 전체 검색량', 'sum'),
        total_inflow=('우리 블로그 검색 유입량', 'sum')
    ).reset_index()

    post_meta_df.rename(columns={'post_id': 'postId'}, inplace=True)
    post_meta_df['postId'] = post_meta_df['postId'].astype(str)
    post_query_df['postId'] = post_query_df['postId'].astype(str)

    master_df = pd.merge(labeled_df, performance_agg_df, on='searchQuery', how='left')
    master_df = pd.merge(master_df, post_query_df[['postId', 'searchQuery']], on='searchQuery', how='left')
    master_df = pd.merge(master_df, post_meta_df[['postId', 'post_title', 'total_views']], on='postId', how='left')
    
    master_df[['total_search_volume', 'total_inflow', 'total_views']] = master_df[['total_search_volume', 'total_inflow', 'total_views']].fillna(0)
    master_df['postId'] = master_df['postId'].fillna('N/A')

    def safe_literal_eval(val):
        try: return ast.literal_eval(val)
        except (ValueError, SyntaxError): return []
        
    master_df['labels_list'] = master_df['labels(json)'].apply(safe_literal_eval)
    master_df['interest_areas'] = master_df['interest_areas'].apply(safe_literal_eval)
    
    # '미분류' 원인 진단
    unclassified_count = master_df['interest_areas'].apply(lambda x: not x).sum()
    total_queries = len(master_df)
    unclassified_ratio = (unclassified_count / total_queries) * 100 if total_queries > 0 else 0
    logging.info(f"'interest_areas'가 비어있어 '미분류'로 처리된 검색어: {unclassified_count}/{total_queries} ({unclassified_ratio:.2f}%)")

    master_df['interest_areas_list'] = master_df['interest_areas'].apply(lambda x: x if x else ['미분류'])

    logging.info("데이터 통합 및 전처리 완료.")
    return master_df, performance_df, num_months, unclassified_ratio

def analyze_topic_1(master_df, num_months):
    """1. 지역 시장 세분화 및 타겟팅 전략 수립 (월 평균 기준)"""
    logging.info("[분석 1] 지역 시장 세분화 분석을 수행합니다.")
    
    local_labels_df = master_df.copy()
    local_labels_df['local_label'] = local_labels_df['labels_list'].apply(
        lambda labels: [l['label'].split(':')[1] for l in labels if l['label'].startswith('지역/장소:')]
    )
    local_labels_df = local_labels_df.explode('local_label')
    local_labels_df = local_labels_df[local_labels_df['local_label'].notna()]

    local_keyword_performance = local_labels_df.groupby('local_label').agg(
        total_search_volume=('total_search_volume', 'sum'),
        total_inflow=('total_inflow', 'sum'),
        related_query_count=('searchQuery', 'nunique')
    ).reset_index()

    if num_months > 0:
        local_keyword_performance['avg_monthly_search_volume'] = (local_keyword_performance['total_search_volume'] / num_months).round(0)
        local_keyword_performance['avg_monthly_inflow'] = (local_keyword_performance['total_inflow'] / num_months).round(0)
    else:
        local_keyword_performance['avg_monthly_search_volume'] = 0
        local_keyword_performance['avg_monthly_inflow'] = 0

    local_keyword_performance['market_share'] = (local_keyword_performance['total_inflow'] / local_keyword_performance['total_search_volume']).fillna(0) * 100
    topic1_1_result = local_keyword_performance[[
        'local_label', 'avg_monthly_search_volume', 'avg_monthly_inflow', 'related_query_count', 'market_share'
    ]].sort_values(by='avg_monthly_inflow', ascending=False)
    topic1_1_result['market_share'] = topic1_1_result['market_share'].map('{:,.2f}%'.format)

    local_search_df = master_df[master_df['labels_list'].apply(lambda labels: any(l['label'].startswith('지역/장소:') for l in labels))]
    local_search_df = local_search_df.explode('interest_areas_list')
    topic1_2_result = local_search_df.groupby('interest_areas_list').agg(
        total_search_volume=('total_search_volume', 'sum'),
        total_inflow=('total_inflow', 'sum')
    ).reset_index()
    
    if num_months > 0:
        topic1_2_result['avg_monthly_search_volume'] = (topic1_2_result['total_search_volume'] / num_months).round(0)
        topic1_2_result['avg_monthly_inflow'] = (topic1_2_result['total_inflow'] / num_months).round(0)
    else:
        topic1_2_result['avg_monthly_search_volume'] = 0
        topic1_2_result['avg_monthly_inflow'] = 0

    topic1_2_result['market_share'] = (topic1_2_result['total_inflow'] / topic1_2_result['total_search_volume']).fillna(0) * 100
    topic1_2_result = topic1_2_result[[
        'interest_areas_list', 'avg_monthly_search_volume', 'avg_monthly_inflow', 'market_share'
    ]].sort_values(by='avg_monthly_search_volume', ascending=False)
    topic1_2_result['market_share'] = topic1_2_result['market_share'].map('{:,.2f}%'.format)
    
    return topic1_1_result, topic1_2_result

def analyze_topic_2(master_df, performance_df, output_dir):
    """2. 브랜딩 퍼널(Funnel) 진단 및 최적화 (이중 축 그래프)"""
    logging.info("[분석 2] 브랜딩 퍼널 진단 분석을 수행합니다.")
    
    master_df['postId'] = master_df['postId'].astype(str).replace('\.0', '', regex=True)
    topic2_1_result = master_df.groupby(['postId', 'post_title', 'stage'])['total_inflow'].sum().reset_index()
    top_posts_stage4 = topic2_1_result[topic2_1_result['stage'] == '4단계:병원인지'].nlargest(5, 'total_inflow')
    top_posts_stage23 = topic2_1_result[topic2_1_result['stage'].isin(['2단계:정보탐색', '3단계:지역탐색'])].nlargest(5, 'total_inflow')

    stage_monthly_data = pd.merge(performance_df, master_df[['searchQuery', 'stage']], on='searchQuery', how='left')
    stage_monthly_agg = stage_monthly_data.groupby(['date', 'stage']).agg(
        total_inflow=('우리 블로그 검색 유입량', 'sum'),
        total_search_volume=('네이버 전체 검색량', 'sum')
    ).unstack(level='stage').fillna(0)

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Bar chart for our blog inflow (Left Y-axis)
    inflow_data = stage_monthly_agg['total_inflow']
    bar_width = 10
    ax1.bar(inflow_data.index - pd.DateOffset(days=bar_width/2), inflow_data.get('3단계:지역탐색', 0), width=bar_width, color='skyblue', alpha=0.7, label='(좌) 우리 블로그 유입량 (3단계: 지역)')
    ax1.bar(inflow_data.index + pd.DateOffset(days=bar_width/2), inflow_data.get('4단계:병원인지', 0), width=bar_width, color='royalblue', alpha=0.7, label='(좌) 우리 블로그 유입량 (4단계: 병원)')
    ax1.set_xlabel('월')
    ax1.set_ylabel('우리 블로그 유입량 (건)', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Line chart for total market search volume (Right Y-axis)
    ax2 = ax1.twinx()
    search_volume_data = stage_monthly_agg['total_search_volume']
    ax2.plot(search_volume_data.index, search_volume_data.get('3단계:지역탐색', 0), color='mediumseagreen', marker='o', linestyle='--', label='(우) 전체 검색량 (3단계: 지역)')
    ax2.plot(search_volume_data.index, search_volume_data.get('4단계:병원인지', 0), color='darkgreen', marker='s', linestyle='--', label='(우) 전체 검색량 (4단계: 병원)')
    ax2.set_ylabel('네이버 전체 검색량 (건)', color='darkgreen')
    ax2.tick_params(axis='y', labelcolor='darkgreen')

    fig.suptitle('시장 검색량(선) vs 우리 블로그 유입량(막대) 추이', fontsize=16)
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(output_dir, 'stage_inflow_trend_combined.png')
    plt.savefig(plot_path)
    logging.info(f"결합된 단계별 유입량 추이 그래프 저장: {plot_path}")
    
    return top_posts_stage4, top_posts_stage23, plot_path

def analyze_topic_3(master_df):
    """3. 콘텐츠 전략 재설계 및 검색 의도 분석"""
    logging.info("[분석 3] 콘텐츠 및 검색의도 분석을 수행합니다.")
    
    # 3-1. 포스트별 실질적 '주제' 분석
    post_interest_df = master_df.explode('interest_areas_list')
    post_interest_inflow = post_interest_df.groupby(['postId', 'post_title', 'interest_areas_list'])['total_inflow'].sum().reset_index()
    dominant_interest = post_interest_inflow.loc[post_interest_inflow.groupby('postId')['total_inflow'].idxmax()]
    topic3_1_result = dominant_interest.sort_values(by='total_inflow', ascending=False).head(10)

    # 3-2. 검색어 카테고리/라벨별 심층 분석
    labels_df = master_df.explode('labels_list')
    labels_df.dropna(subset=['labels_list'], inplace=True)
    labels_df['category'] = labels_df['labels_list'].apply(lambda x: x['label'].split(':')[0] if isinstance(x, dict) and ':' in x['label'] else 'N/A')
    labels_df['label_name'] = labels_df['labels_list'].apply(lambda x: x['label'].split(':')[1] if isinstance(x, dict) and ':' in x['label'] else 'N/A')

    category_performance = labels_df.groupby('category').agg(
        total_inflow=('total_inflow', 'sum'),
        related_query_count=('searchQuery', 'nunique')
    ).sort_values('total_inflow', ascending=False).reset_index()

    intent_labels = labels_df[labels_df['category'] == '검색의도']
    label_performance = intent_labels.groupby('label_name').agg(
        total_inflow=('total_inflow', 'sum'),
        related_query_count=('searchQuery', 'nunique')
    ).sort_values('total_inflow', ascending=False).reset_index()

    return topic3_1_result, category_performance, label_performance

def generate_html_report(results, output_path, plot_path, unclassified_ratio):
    """분석 결과를 종합하여 동적인 HTML 보고서를 생성합니다."""
    logging.info("HTML 보고서 생성을 시작합니다.")
    (topic1_1, topic1_2), (topic2_1_s4, topic2_1_s23), (topic3_1, topic3_2_cat, topic3_2_label) = results

    # DataFrames to HTML tables
    topic1_1_html = topic1_1.to_html(index=False, classes='data-table', border=0)
    topic1_2_html = topic1_2.to_html(index=False, classes='data-table', border=0)
    topic2_1_s4_html = topic2_1_s4.to_html(index=False, classes='data-table', border=0)
    topic2_1_s23_html = topic2_1_s23.to_html(index=False, classes='data-table', border=0)
    topic3_1_html = topic3_1.to_html(index=False, classes='data-table', border=0)
    topic3_2_cat_html = topic3_2_cat.to_html(index=False, classes='data-table', border=0)
    topic3_2_label_html = topic3_2_label.to_html(index=False, classes='data-table', border=0)

    # f-string에서 중괄호 문제를 피하기 위해 스타일과 스크립트를 별도 문자열로 분리
    style_section = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
    :root {
        --primary-color: #005AEE; --secondary-color: #4A90E2; --accent-color: #9f7aea;
        --text-color: #2d3748; --text-light-color: #4a5568; --bg-color: #f9fafb;
        --white-color: #ffffff; --grey-color: #e2e8f0; --dark-grey-color: #a0aec0;
    }
    body {
        font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.8; color: var(--text-color); background-color: var(--bg-color);
        margin: 0; padding: 0;
    }
    .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
    .header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        color: var(--white-color); padding: 60px 40px; border-radius: 20px; margin-bottom: 40px;
        text-align: center; box-shadow: 0 10px 30px rgba(0, 90, 238, 0.2);
    }
    h1 { margin: 0; font-size: 3em; font-weight: 800; letter-spacing: -1px; }
    .subtitle { margin-top: 15px; font-size: 1.3em; opacity: 0.95; font-weight: 300; }
    .tab-container { margin: 30px 0; }
    .tab-buttons { display: flex; border-bottom: 3px solid var(--grey-color); }
    .tab-button {
        flex: 1; padding: 16px 24px; background: none; border: none; cursor: pointer;
        font-weight: 700; font-size: 1.1em; color: #555; transition: all 0.3s ease;
        border-bottom: 4px solid transparent; margin-bottom: -3px;
    }
    .tab-button:hover { background-color: #f0f0f0; }
    .tab-button.active {
        color: var(--white-color); background-color: var(--primary-color);
        border-bottom-color: var(--secondary-color); border-radius: 12px 12px 0 0;
    }
    .tab-content {
        display: none; background: var(--white-color); padding: 40px; margin-top: -1px;
        border: 1px solid var(--grey-color); border-top: none;
        border-radius: 0 0 16px 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        animation: fadeIn 0.5s;
    }
    .tab-content.active { display: block; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    h2 {
        color: var(--primary-color); border-bottom: 4px solid var(--primary-color);
        padding-bottom: 15px; margin-bottom: 30px; font-size: 2.2em;
        font-weight: 700; display: flex; align-items: center; gap: 15px;
    }
    h3 {
        color: var(--text-light-color); margin-top: 40px; margin-bottom: 20px; font-size: 1.6em;
        background: var(--bg-color); padding: 15px 20px; border-left: 5px solid var(--accent-color);
        border-radius: 8px; font-weight: 700;
    }
    .info-box, .success-box, .warning-box, .danger-box {
        padding: 20px; margin: 20px 0; border-radius: 8px;
        border-left-width: 5px; border-left-style: solid; font-weight: 500;
    }
    .info-box { background: #ebf8ff; border-color: #4299e1; }
    .success-box { background: #f0fff4; border-color: #48bb78; }
    .warning-box { background: #fffaf0; border-color: #f6ad55; }
    .danger-box { background: #fff5f5; border-color: #f56565; }
    .roadmap-table {
        overflow-x: auto; margin: 25px 0; border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05); border: 1px solid var(--grey-color);
    }
    table.data-table { width: 100%; border-collapse: collapse; background: var(--white-color); }
    table.data-table th {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: var(--white-color); padding: 15px; text-align: left;
        font-weight: 600; font-size: 0.9em; letter-spacing: 0.5px;
    }
    table.data-table td {
        padding: 15px; border-bottom: 1px solid var(--grey-color);
        vertical-align: top; font-size: 0.95em;
    }
    table.data-table tr:hover { background-color: #f7fafc; }
    table.data-table tr:last-child td { border-bottom: none; }
</style>
"""

    script_section = """
<script>
    function openTab(evt, tabName) {
        var i, tabcontent, tabbuttons;
        tabcontent = document.getElementsByClassName("tab-content");
        for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
            tabcontent[i].classList.remove("active");
        }
        tabbuttons = document.getElementsByClassName("tab-button");
        for (i = 0; i < tabbuttons.length; i++) {
            tabbuttons[i].classList.remove("active");
        }
        document.getElementById(tabName).style.display = "block";
        document.getElementById(tabName).classList.add("active");
        if (evt) {
            evt.currentTarget.classList.add("active");
        }
    }
    document.addEventListener("DOMContentLoaded", function() {
        const firstTabButton = document.querySelector(".tab-button");
        if (firstTabButton) {
            firstTabButton.click();
        }
    });
</script>
"""

    html_body = f"""
<body>
<div class="container">
    <div class="header">
        <h1>블로그 성과 분석 보고서 (v2.0)</h1>
        <div class="subtitle">데이터 기반 마케팅 전략 제안</div>
    </div>

    <div class="tab-container">
        <div class="tab-buttons">
            <button class="tab-button" onclick="openTab(event, 'tab-summary')">📊 요약 및 제안</button>
            <button class="tab-button" onclick="openTab(event, 'tab-local')">🗺️ 지역/진료 분석</button>
            <button class="tab-button" onclick="openTab(event, 'tab-funnel')">🚦 브랜딩 퍼널 진단</button>
            <button class="tab-button" onclick="openTab(event, 'tab-content')">📝 콘텐츠/의도 분석</button>
        </div>

        <div id="tab-summary" class="tab-content">
            <h2>📊 분석 요약 및 핵심 제안</h2>
            <div class="info-box">
                <strong>본 보고서는 블로그 유입 검색어, 성과 데이터, 포스트 데이터를 종합하여 우리 병원의 현재 마케팅 포지션을 진단하고, 데이터 기반의 구체적인 실행 전략을 제안합니다.</strong>
            </div>
            <h3>핵심 발견 (Key Findings)</h3>
            <p><strong>- '동탄역' 키워드의 높은 효율성:</strong> '동탄' 전체보다 '동탄역' 키워드의 시장 점유율이 월등히 높아, 해당 지역 타겟팅이 매우 유효함을 확인했습니다.</p>
            <p><strong>- 시장 트렌드와 유입량의 동반 성장:</strong> 전체적인 시장의 검색량(3, 4단계)과 우리 블로그 유입량이 동반 성장하는 긍정적인 추세를 보이고 있습니다. 이는 우리의 마케팅 활동이 시장의 관심사와 잘 부합하고 있음을 시사합니다.</p>
            <p><strong>- '비용/보험', '후기' 검색 의도의 중요성:</strong> 블로그 유입의 상당수가 '비용'과 '후기' 관련 검색 의도를 통해 발생하고 있어, 이 두 가지 주제가 고객의 핵심적인 궁금증임을 재확인했습니다.</p>
            
            <h3>핵심 제안 (Action Items)</h3>
            <p><strong>- '동탄역' + '충치/신경치료' 콘텐츠 강화:</strong> 가장 효율이 높은 '동탄역' 키워드와 시장 규모가 가장 큰 '충치', '신경치료'를 조합한 콘텐츠를 집중적으로 기획해야 합니다. ('동탄역 충치치료 비용', '동탄역 신경치료 후기' 등)</p>
            <p><strong>- '비용/보험' 및 '후기' 콘텐츠 포맷 다각화:</strong> 고객의 핵심 궁금증에 답하기 위해, '진료별 비용 총정리', '환자 케이스별 치료 과정 및 후기' 등 신뢰도를 높일 수 있는 포맷의 콘텐츠를 강화해야 합니다.</p>
        </div>

        <div id="tab-local" class="tab-content">
            <h2>🗺️ 분석 1: 지역 시장 및 핵심 진료 분야 분석</h2>
            <h3>1-1. '효자 지역 키워드' 월 평균 성과</h3>
            <div class="info-box"><strong>Objective:</strong> 어떤 지역 키워드가 우리 블로그 유입에 가장 효과적인지 파악하여, 지역 타겟팅의 우선순위를 설정합니다. (월 평균 기준)</div>
            <div class="success-box"><strong>Finding:</strong> '동탄' 키워드가 월 평균 가장 많은 유입을 만들고 있지만, '시장 점유율' 측면에서는 '화성', '로덴치과' 등 더 작은 단위의 키워드들이 높은 효율을 보입니다. '동탄역' 역시 전체 검색량 대비 높은 점유율을 기록하며 중요 타겟임을 보여줍니다.</div>
            <div class="roadmap-table">{topic1_1_html}</div>
            <div class="warning-box"><strong>Action Item:</strong> '동탄역', '오산동' 등 상대적으로 높은 점유율을 보이는 세부 지역 키워드를 활용하여, '동탄역 직장인 임플란트'와 같이 더 구체적이고 타겟팅된 콘텐츠를 기획하여 틈새시장을 공략할 필요가 있습니다.</div>

            <h3>1-2. 핵심 진료 분야별 '지역(Local)' 월 평균 시장 점유율</h3>
            <div class="info-box"><strong>Objective:</strong> 우리 병원의 핵심 타겟 시장인 '지역 내'에서 어떤 진료 분야에 강점과 약점이 있는지 월 평균 기준으로 진단합니다.</div>
            <div class="success-box"><strong>Finding:</strong> '충치치료', '신경치료'가 월 평균 검색량 기준 가장 큰 시장을 형성하고 있습니다. '심미치료'는 검색량은 적지만 매우 높은 시장 점유율을 보여, 해당 분야에서 우리 블로그가 강력한 영향력을 가지고 있음을 의미합니다.</div>
            <div class="roadmap-table">{topic1_2_html}</div>
            <div class="warning-box"><strong>Action Item:</strong> 시장이 가장 큰 '충치치료'와 '신경치료' 분야의 지역 내 점유율을 높이는 것이 가장 시급한 과제입니다. '동탄 충치치료 잘하는 곳', '동탄역 신경치료 후기'와 같이 [지역명 + 진료명 + 검색의도] 조합의 롱테일 키워드 콘텐츠를 집중적으로 발행하여 시장 점유율을 끌어올려야 합니다.</div>
        </div>

        <div id="tab-funnel" class="tab-content">
            <h2>🚦 분석 2: 브랜딩 퍼널(Funnel) 진단 및 최적화</h2>
            <h3>2-1. '충성 고객' vs '신규 고객' 유입 채널 분석</h3>
            <div class="info-box"><strong>Objective:</strong> 우리 블로그의 포스트들이 브랜드 인지 고객 유지(충성 고객)와 신규 고객 유치 역할을 각각 얼마나 잘 수행하는지 진단합니다.</div>
            <div class="success-box">
                <strong>Finding:</strong><br>
                - <strong>4단계(병원 인지) 고객 유입 상위 포스트 (브랜드 유지 역할):</strong>
                <div class="roadmap-table">{topic2_1_s4_html}</div>
                - <strong>2/3단계(정보/지역 탐색) 고객 유입 상위 포스트 (신규 고객 유치 역할):</strong>
                <div class="roadmap-table">{topic2_1_s23_html}</div>
            </div>
            <div class="warning-box"><strong>Action Item:</strong> 신규 고객 유치 포스트와 브랜드 유지 포스트를 내부 링크로 연결하여, 우리를 처음 알게 된 고객이 자연스럽게 우리 병원에 대한 신뢰를 쌓고 충성 고객으로 전환될 경로를 설계해야 합니다. (예: '이가 깨졌어요' 포스트 말미에 '내이튼치과의 앞니 레진 치료 실제 후기' 포스트 링크 추가)</div>

            <h3>2-2. 시장 검색량(선) vs 우리 블로그 유입량(막대) 추이</h3>
            <div class="info-box"><strong>Objective:</strong> 시장의 전체 검색량 트렌드와 우리 블로그의 유입 성과를 비교하여, 마케팅 활동의 효과를 입체적으로 분석합니다.</div>
            <div class="success-box"><strong>Finding:</strong><br><img src="{plot_path}" alt="단계별 유입량 추이" style="width:100%; max-width:1000px; margin:auto; display:block;"></div>
            <div class="warning-box"><strong>Action Item:</strong> 녹색 선(시장 전체 검색량)이 상승하는 시기에 파란색 막대(우리 블로그 유입량)도 함께 상승하는 것은 매우 긍정적인 신호입니다. 만약 녹색 선은 오르는데 파란 막대가 정체된다면, 시장의 관심을 우리 블로그로 가져오지 못하고 있다는 의미이므로 해당 월의 포스트 전략을 재검토해야 합니다.</div>
        </div>

        <div id="tab-content" class="tab-content">
            <h2>📝 분석 3: 콘텐츠 및 검색 의도 분석</h2>
            <h3>3-1. 포스트별 실질적 '주제' 분석</h3>
            <div class="info-box"><strong>Objective:</strong> 각 포스트가 실제로 어떤 주제(관심 진료 분야)로 시장에서 인식되고 있는지 파악하여, 콘텐츠의 역할을 재정의하거나 개선 방향을 설정합니다.</div>
            <div class="danger-box"><strong>진단 필요:</strong> '관심 진료 분야(interest_areas)' 데이터의 <strong>{unclassified_ratio:.2f}%</strong>가 '미분류' 상태입니다. 이는 원본 데이터 생성 단계에서 대부분의 검색어에 대한 관심 진료 분야가 추출되지 않았음을 의미합니다. 정확한 분석을 위해 선행 `apply_restructured_taxonomy.py` 스크립트의 LLM 기반 'interest_areas' 추출 로직 점검이 필요합니다.</div>
            <div class="success-box"><strong>Finding:</strong> 현재 데이터 기준, 아래는 각 포스트별로 가장 많은 유입을 발생시킨 관심 진료 분야입니다. 대부분 '미분류'로 나타나고 있어, 아래 결과는 제한적인 정보만을 제공합니다.
            <div class="roadmap-table">{topic3_1_html}</div>
            </div>

            <h3>3-2. 검색 의도 심층 분석</h3>
            <div class="info-box"><strong>Objective:</strong> 사용자들이 어떤 의도(카테고리 및 라벨)를 가지고 검색하며, 이를 통해 우리 블로그로 유입되는지 파악합니다.</div>
            <div class="success-box">
                <strong>Finding 1: 카테고리별 유입 성과</strong><br>
                '신체부위' 카테고리가 가장 많은 유입을 발생시키고 있으며, '검색의도'와 '증상/상태'가 그 뒤를 잇고 있습니다. 이는 고객들이 특정 치아나 잇몸의 문제에 대해 구체적인 해결책(비용, 후기 등)을 찾으려는 경향이 강함을 보여줍니다.
                <div class="roadmap-table">{topic3_2_cat_html}</div>
                <strong>Finding 2: '검색의도' 카테고리 내 라벨별 유입 성과</strong><br>
                가장 중요한 '검색의도' 카테고리 내에서는 '비용/보험'과 '병원/의사 추천' (후기, 잘하는 곳 등)에 대한 관심이 압도적으로 높습니다.
                <div class="roadmap-table">{topic3_2_label_html}</div>
            </div>
            <div class="warning-box"><strong>Action Item:</strong> '비용/보험', '병원/의사 추천' 두 핵심 의도에 직접적으로 답변하는 콘텐츠를 시리즈로 기획해야 합니다. 예를 들어, '동탄 임플란트 가격, 보험 적용 시 실제 비용은?' 또는 '실제 환자가 말하는 내이튼치과 신경치료 후기'와 같은 제목의 콘텐츠는 높은 유입을 기대할 수 있습니다.</div>
        </div>
    </div>
</div>
</body>
"""

    html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Medicontents: 블로그 성과 분석 보고서 v2.0</title>
{style_section}
</head>
{html_body}
{script_section}
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    logging.info(f"HTML 보고서 저장 완료: {output_path}")


def generate_markdown_report(results, output_path, plot_path, unclassified_ratio):
    """분석 결과를 종합하여 마크다운 보고서를 생성합니다."""
    logging.info("마크다운 보고서 생성을 시작합니다.")
    (topic1_1, topic1_2), (topic2_1_s4, topic2_1_s23), (topic3_1, topic3_2_cat, topic3_2_label) = results

    report = f"""# 마케팅 성과 분석 및 전략 제안 보고서 (v2.0)

최종 분석 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 분석 요약 및 핵심 제안 (Executive Summary)

- **핵심 발견 1: '동탄역' 키워드의 높은 효율성.** '동탄' 전체보다 '동탄역' 키워드의 월 평균 시장 점유율이 월등히 높아, 해당 지역 타겟팅이 매우 유효함을 확인했습니다.
- **핵심 발견 2: 시장 트렌드와 유입량의 동반 성장.** 전체적인 시장의 검색량(3, 4단계)과 우리 블로그 유입량이 동반 성장하는 긍정적인 추세를 보이고 있습니다.
- **핵심 발견 3: '비용/보험', '후기' 검색 의도의 중요성.** 블로그 유입의 상당수가 '비용'과 '후기' 관련 검색 의도를 통해 발생하고 있어, 이 두 가지 주제가 고객의 핵심적인 궁금증임을 재확인했습니다.
- **핵심 제안:** '동탄역'과 같은 효율 높은 지역 키워드와 시장이 큰 '충치/신경치료'를 조합하고, 고객의 핵심 궁금증인 '비용/후기'에 대한 답을 주는 콘텐츠를 강화해야 합니다.

---

## 분석 1: 지역 시장 및 핵심 진료 분야 분석

### 1-1. '효자 지역 키워드' 월 평균 성과
**Objective:** 어떤 지역 키워드가 우리 블로그 유입에 가장 효과적인지 월 평균 기준으로 파악합니다.

**Finding:**
'동탄' 키워드가 월 평균 가장 많은 유입을 만들고 있지만, '시장 점유율' 측면에서는 '화성', '로덴치과' 등 더 작은 단위의 키워드들이 높은 효율을 보입니다. '동탄역' 역시 전체 검색량 대비 높은 점유율을 기록하며 중요 타겟임을 보여줍니다.

{topic1_1.to_markdown(index=False)}

### 1-2. 핵심 진료 분야별 '지역(Local)' 월 평균 시장 점유율
**Objective:** 우리 병원의 핵심 타겟 시장인 '지역 내'에서 어떤 진료 분야에 강점과 약점이 있는지 월 평균 기준으로 진단합니다.

**Finding:**
'충치치료', '신경치료'가 월 평균 검색량 기준 가장 큰 시장을 형성하고 있습니다. '심미치료'는 검색량은 적지만 매우 높은 시장 점유율을 보여, 해당 분야에서 우리 블로그가 강력한 영향력을 가지고 있음을 의미합니다.

{topic1_2.to_markdown(index=False)}

---

## 분석 2: 브랜딩 퍼널(Funnel) 진단 및 최적화

### 2-1. '충성 고객' vs '신규 고객' 유입 채널 분석
**Objective:** 우리 블로그의 포스트들이 브랜드 인지 고객 유지(충성 고객)와 신규 고객 유치 역할을 각각 얼마나 잘 수행하는지 진단합니다.

**Finding:**
- **4단계(병원 인지) 고객 유입 상위 포스트 (브랜드 유지 역할):**
{topic2_1_s4.to_markdown(index=False)}

- **2/3단계(정보/지역 탐색) 고객 유입 상위 포스트 (신규 고객 유치 역할):**
{topic2_1_s23.to_markdown(index=False)}

### 2-2. 시장 검색량(선) vs 우리 블로그 유입량(막대) 추이
**Objective:** 시장의 전체 검색량 트렌드와 우리 블로그의 유입 성과를 비교하여, 마케팅 활동의 효과를 입체적으로 분석합니다.

**Finding:**
![단계별 유입량 추이]({os.path.basename(plot_path)})

**Action Item:**
녹색 선(시장 전체 검색량)이 상승하는 시기에 파란색 막대(우리 블로그 유입량)도 함께 상승하는 것은 매우 긍정적인 신호입니다. 만약 녹색 선은 오르는데 파란 막대가 정체된다면, 시장의 관심을 우리 블로그로 가져오지 못하고 있다는 의미이므로 해당 월의 포스트 전략을 재검토해야 합니다.

---

## 분석 3: 콘텐츠 및 검색 의도 분석

### 3-1. 포스트별 실질적 '주제' 분석
**Objective:** 각 포스트가 실제로 어떤 주제(관심 진료 분야)로 시장에서 인식되고 있는지 파악합니다.

**진단 필요:**
'관심 진료 분야(interest_areas)' 데이터의 **{unclassified_ratio:.2f}%**가 '미분류' 상태입니다. 이는 원본 데이터 생성 단계에서 대부분의 검색어에 대한 관심 진료 분야가 추출되지 않았음을 의미합니다. 정확한 분석을 위해 선행 `apply_restructured_taxonomy.py` 스크립트의 LLM 기반 'interest_areas' 추출 로직 점검이 필요합니다.

**Finding:**
현재 데이터 기준, 아래는 각 포스트별로 가장 많은 유입을 발생시킨 관심 진료 분야입니다. 대부분 '미분류'로 나타나고 있어, 아래 결과는 제한적인 정보만을 제공합니다.

{topic3_1.to_markdown(index=False)}

### 3-2. 검색 의도 심층 분석 (신규)
**Objective:** 사용자들이 어떤 의도(카테고리 및 라벨)를 가지고 검색하며, 이를 통해 우리 블로그로 유입되는지 파악합니다.

**Finding 1: 카테고리별 유입 성과**
'신체부위' 카테고리가 가장 많은 유입을 발생시키고 있으며, '검색의도'와 '증상/상태'가 그 뒤를 잇고 있습니다. 이는 고객들이 특정 치아나 잇몸의 문제에 대해 구체적인 해결책(비용, 후기 등)을 찾으려는 경향이 강함을 보여줍니다.

{topic3_2_cat.to_markdown(index=False)}

**Finding 2: '검색의도' 카테고리 내 라벨별 유입 성과**
가장 중요한 '검색의도' 카테고리 내에서는 '비용/보험'과 '병원/의사 추천' (후기, 잘하는 곳 등)에 대한 관심이 압도적으로 높습니다.

{topic3_2_label.to_markdown(index=False)}

**Action Item:**
'비용/보험', '병원/의사 추천' 두 핵심 의도에 직접적으로 답변하는 콘텐츠를 시리즈로 기획해야 합니다. 예를 들어, '동탄 임플란트 가격, 보험 적용 시 실제 비용은?' 또는 '실제 환자가 말하는 내이튼치과 신경치료 후기'와 같은 제목의 콘텐츠는 높은 유입을 기대할 수 있습니다.
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logging.info(f"마크다운 보고서 저장 완료: {output_path}")


def main():
    """메인 실행 함수"""
    logging.info("===== 통합 마케팅 분석 및 리포팅 시스템 시작 (v2.0) =====")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_path, 'data/data_processed/0728_2300')
    os.makedirs(output_dir, exist_ok=True)
    
    master_df, performance_df, num_months, unclassified_ratio = load_and_merge_data(base_path)
    
    if master_df is None:
        logging.error("데이터 로딩 실패로 분석을 중단합니다.")
        return

    # 분석 수행
    topic1_results = analyze_topic_1(master_df, num_months)
    topic2_results = analyze_topic_2(master_df, performance_df, output_dir)
    topic3_results = analyze_topic_3(master_df)

    # 보고서 생성
    report_path_html = os.path.join(output_dir, 'marketing_insights_report.html')
    report_path_md = os.path.join(output_dir, 'marketing_insights_report.md')
    plot_path_relative = os.path.basename(topic2_results[2])
    
    # HTML 보고서
    generate_html_report(
        (topic1_results, topic2_results[0:2], topic3_results),
        report_path_html,
        plot_path_relative,
        unclassified_ratio
    )
    
    # 마크다운 보고서
    generate_markdown_report(
        (topic1_results, topic2_results[0:2], topic3_results),
        report_path_md,
        plot_path_relative,
        unclassified_ratio
    )

    logging.info("===== 모든 분석 및 보고서 생성이 성공적으로 완료되었습니다. =====")


if __name__ == "__main__":
    main() 