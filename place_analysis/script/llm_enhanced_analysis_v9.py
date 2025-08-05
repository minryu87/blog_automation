import pandas as pd
import numpy as np
import os
import glob
import warnings
import json
import re
from dotenv import load_dotenv, find_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64
from markdown2 import Markdown

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# --- Matplotlib Configuration for Korean Fonts ---
import matplotlib.font_manager as fm
import platform
import subprocess

def find_korean_fonts():
    korean_fonts = []
    if platform.system() == 'Darwin':
        font_paths = ['/System/Library/Fonts/Supplemental/AppleGothic.ttf']
        for font_path in font_paths:
            if os.path.exists(font_path):
                korean_fonts.append(font_path)
        try:
            result = subprocess.run(['fc-list', ':lang=ko'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        font_path = line.split(':')[0]
                        korean_fonts.append(font_path)
        except:
            pass
    return korean_fonts

font_found = False
korean_font_paths = find_korean_fonts()
if korean_font_paths:
    for font_path in korean_font_paths:
        try:
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False
            print(f"한글 폰트 '{font_name}' 설정 완료. (경로: {font_path})")
            font_found = True
            break
        except Exception as e:
            print(f"폰트 '{font_path}' 설정 실패: {e}")
            continue

if not font_found:
    korean_font_names = ['Apple SD Gothic Neo', 'AppleGothic', 'Arial Unicode MS', 'NanumGothic', 'Malgun Gothic']
    for font_name in korean_font_names:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and ('ttf' in font_path.lower() or 'otf' in font_path.lower() or 'ttc' in font_path.lower()):
                plt.rc('font', family=font_name)
                plt.rcParams['axes.unicode_minus'] = False
                print(f"한글 폰트 '{font_name}' 설정 완료.")
                font_found = True
                break
        except Exception as e:
            print(f"폰트 '{font_name}' 설정 실패: {e}")
            continue

if not font_found:
    print("경고: 적절한 한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

# --- Configuration ---
RAW_DATA_DIR = 'blog_automation/place_analysis/data/raw_data/동탄치과'
RESULT_PATH = 'blog_automation/place_analysis/analysis_result'
os.makedirs(RESULT_PATH, exist_ok=True)
REPORT_OUTPUT_PATH_HTML = os.path.join(RESULT_PATH, 'llm_enhanced_analysis_v9.4_report.html')
TARGET_CLINIC = '내이튼치과의원'

class RankAnalyzerV9:
    def __init__(self, data_dir, result_path):
        self.data_dir = data_dir
        self.result_path = result_path
        self.agent = self._initialize_llm_agent()
        self.time_series_df = self._load_and_prepare_time_series_data()
        self.markdown_converter = Markdown()

    def _initialize_llm_agent(self):
        print("LLM 에이전트 초기화 중...")
        load_dotenv(find_dotenv())
        try:
            llm = Gemini(id=os.getenv("GEMINI_MODEL", "gemini/gemini-1.5-pro-latest"), api_key=os.getenv("GEMINI_API_KEY"))
            return Agent(model=llm)
        except Exception as e:
            print(f"LLM 에이전트 초기화 실패: {e}.")
            return None

    def _md_to_df_string(self, data):
        return {k: (v.to_markdown(index=False) if isinstance(v, pd.DataFrame) and not v.empty else v) for k, v in data.items()}

    def _load_and_prepare_time_series_data(self):
        csv_files = glob.glob(os.path.join(self.data_dir, 'vs*.csv'))
        if not csv_files: return pd.DataFrame()
        all_data = []
        df_t0 = pd.read_csv(os.path.join(self.data_dir, 'vs1.csv'))
        df_t0['company_name'] = df_t0['company_name_category'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else None)
        df_t0.dropna(subset=['company_name'], inplace=True)
        t0_data = df_t0[['company_name', 'rank', 'visitor_reviews', 'blog_reviews']].copy()
        t0_data['days_ago'] = 0
        all_data.append(t0_data)
        for f in csv_files:
            try:
                if 'vs1.csv' in f: continue
                days_ago = int(re.search(r'vs(\d+)\.csv', f).group(1))
                df = pd.read_csv(f)
                df['company_name'] = df['company_name_category'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else None)
                df.dropna(subset=['company_name'], inplace=True)
                df['rank_change'] = pd.to_numeric(df['rank_change'], errors='coerce').fillna(0)
                df['rank_trend'] = pd.to_numeric(df['rank_trend'], errors='coerce').fillna(2)
                def calculate_past_rank(row):
                    if row['rank_trend'] == 1: return row['rank'] + row['rank_change']
                    elif row['rank_trend'] == 2: return row['rank']
                    elif row['rank_trend'] == 3: return row['rank'] - row['rank_change']
                    else: return row['rank']
                df['past_rank'] = df.apply(calculate_past_rank, axis=1)
                df_t0_merged = df_t0[['company_name', 'visitor_reviews', 'blog_reviews']].copy()
                merged = pd.merge(df_t0_merged, df[['company_name', 'visitor_reviews_change', 'blog_reviews_change']], on='company_name', how='inner')
                for col in ['visitor_reviews_change', 'blog_reviews_change']:
                    merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)
                merged['visitor_reviews_past'] = merged['visitor_reviews'] - merged['visitor_reviews_change']
                merged['blog_reviews_past'] = merged['blog_reviews'] - merged['blog_reviews_change']
                historical_data = pd.merge(df[['company_name', 'past_rank']], merged[['company_name', 'visitor_reviews_past', 'blog_reviews_past']], on='company_name', how='inner')
                historical_data.rename(columns={'past_rank': 'rank', 'visitor_reviews_past': 'visitor_reviews', 'blog_reviews_past': 'blog_reviews'}, inplace=True)
                historical_data['days_ago'] = days_ago
                all_data.append(historical_data)
            except Exception as e:
                print(f"'{f}' 파일 처리 중 오류 발생: {e}")
        full_df = pd.concat(all_data, ignore_index=True)
        full_df = full_df.drop_duplicates(subset=['company_name', 'days_ago'], keep='last').sort_values(by=['company_name', 'days_ago']).reset_index(drop=True)
        return full_df

    def _create_dynamic_tiers_kmeans(self, df, column_name, n_clusters=5):
        """Creates dynamic tiers using K-Means clustering."""
        print(f"'{column_name}'에 대한 K-평균 클러스터링 기반 동적 등급 생성 중...")
        data = df[[column_name]].dropna()
        if len(data) < n_clusters:
            print(f"데이터 포인트({len(data)}개)가 클러스터 수({n_clusters}개)보다 적어 클러스터 수를 {len(data)}개로 조정합니다.")
            n_clusters = len(data)
        if n_clusters == 0:
            print("분석할 데이터가 없습니다.")
            return df
        if data.empty:
            return df
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(scaled_data)
        df.loc[data.index, f'{column_name}_cluster'] = clusters
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_order = np.argsort(cluster_centers.flatten())[::-1]
        tier_names = [f'Tier {i+1}' for i in range(n_clusters)]
        tier_map = {cluster_order[i]: tier_names[i] for i in range(n_clusters)}
        tier_col_name = f'{column_name}_tier'
        df[tier_col_name] = df[f'{column_name}_cluster'].map(tier_map)
        return df.drop(columns=[f'{column_name}_cluster'])

    def analyze_tier_performance(self, df, tier_col, value_cols):
        """Analyzes performance by tier."""
        value_col = value_cols[0]
        agg_funcs = {value_col: ['size', 'mean', 'std']}
        analysis = df.groupby(tier_col).agg(agg_funcs).reset_index()
        analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]
        analysis.rename(columns={
            f'{tier_col}_': 'tier',
            f'{value_col}_size': '업체 수',
            f'{value_col}_mean': f'{value_col}_mean',
            f'{value_col}_std': f'{value_col}_std'
        }, inplace=True)
        for col in analysis.columns:
             if 'mean' in col or 'std' in col:
                 analysis[col] = analysis[col].fillna(0).round(1)
        return analysis.sort_values(by='tier', ascending=True)

    def analyze_tier_trends(self):
        current_data = self.time_series_df[self.time_series_df['days_ago'] == 0].copy()
        df_with_tiers = self._create_dynamic_tiers_kmeans(current_data.copy(), 'visitor_reviews')
        df_with_tiers = self._create_dynamic_tiers_kmeans(df_with_tiers, 'blog_reviews')
        visit_tier_data = self.analyze_tier_performance(df_with_tiers, 'visitor_reviews_tier', ['visitor_reviews'])
        blog_tier_data = self.analyze_tier_performance(df_with_tiers, 'blog_reviews_tier', ['blog_reviews'])
        past_data = self.time_series_df[self.time_series_df['days_ago'] == 60].copy()
        merged_data = pd.merge(df_with_tiers, past_data, on='company_name', suffixes=('_current', '_past'), how='left')
        merged_data['visitor_reviews_change'] = merged_data['visitor_reviews_current'] - merged_data['visitor_reviews_past'].fillna(merged_data['visitor_reviews_current'])
        merged_data['blog_reviews_change'] = merged_data['blog_reviews_current'] - merged_data['blog_reviews_past'].fillna(merged_data['blog_reviews_current'])
        visit_change_data = self.analyze_tier_performance(merged_data, 'visitor_reviews_tier', ['visitor_reviews_change'])
        blog_change_data = self.analyze_tier_performance(merged_data, 'blog_reviews_tier', ['blog_reviews_change'])
        
        plot_path_visit_tier = os.path.join(self.result_path, 'v9_visit_tier_trends.png')
        self._plot_tier_trends(visit_tier_data, 'visitor_reviews', plot_path_visit_tier)
        plot_path_blog_tier = os.path.join(self.result_path, 'v9_blog_tier_trends.png')
        self._plot_tier_trends(blog_tier_data, 'blog_reviews', plot_path_blog_tier)

        data_for_llm = self._md_to_df_string({
            "visit_tier": visit_tier_data, "blog_tier": blog_tier_data,
            "visit_change": visit_change_data, "blog_change": blog_change_data
        })

        prompt = f"""## 시장 동향 분석 ##
        '동탄치과' 시장의 Tier별 구조와 최근 60일간의 성장 동력에 대해 분석해주세요.
        - 방문자 리뷰와 블로그 리뷰 시장의 구조적 차이점을 설명해주세요.
        - Tier별 리뷰 증가량 데이터를 바탕으로 시장의 경쟁 강도와 '부익부 빈익빈' 현상을 설명해주세요.
        ---
        [방문자 리뷰 Tier 현황]\n{data_for_llm['visit_tier']}
        [블로그 리뷰 Tier 현황]\n{data_for_llm['blog_tier']}
        [방문자 리뷰 Tier별 변화량]\n{data_for_llm['visit_change']}
        [블로그 리뷰 Tier별 변화량]\n{data_for_llm['blog_change']}"""
        
        response = self.agent.run(prompt, max_tokens=8000)
        return {
            "title": "시장 동향 요약: Tier별 시장 구조와 경쟁 현실",
            "content": response.content if response else "",
            "plot_paths": [plot_path_visit_tier, plot_path_blog_tier]
        }

    def _plot_tier_trends(self, tier_data, review_type, file_path):
        type_kor = '블로그 리뷰' if review_type == 'blog_reviews' else '방문자 리뷰'
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        if not tier_data.empty:
            sns.barplot(ax=axes[0], x='tier', y=f'{review_type}_mean', data=tier_data, color='skyblue', hue='tier', legend=False)
            axes[0].set_title(f'{type_kor} Tier별 평균 리뷰 수', fontsize=14, pad=20)
            for index, row in tier_data.iterrows():
                 axes[0].text(index, row[f'{review_type}_mean'], f"{int(row[f'{review_type}_mean']):,}", color='black', ha="center")
            sns.barplot(ax=axes[1], x='tier', y='업체 수', data=tier_data, color='lightcoral', hue='tier', legend=False)
            axes[1].set_title(f'{type_kor} Tier별 업체 수', fontsize=14, pad=20)
            for index, row in tier_data.iterrows():
                axes[1].text(index, row['업체 수'], int(row['업체 수']), color='black', ha="center")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_competitor_landscape(self):
        # Gatekeepers
        df_0 = self.time_series_df[self.time_series_df['days_ago'] == 0].copy()
        df_60 = self.time_series_df[self.time_series_df['days_ago'] == 60].copy()
        top_10_at_60 = df_60[df_60['rank'] <= 10]['company_name']
        top_10_at_0 = df_0[df_0['rank'] <= 10]['company_name']
        gatekeeper_names = pd.merge(top_10_at_60.to_frame(), top_10_at_0.to_frame(), on='company_name')['company_name']
        gatekeepers_ts_data = self.time_series_df[self.time_series_df['company_name'].isin(gatekeeper_names)].copy()
        plot_path_gatekeeper = os.path.join(self.result_path, 'v9_gatekeeper_ranks.png')
        self._plot_gatekeepers(gatekeepers_ts_data, plot_path_gatekeeper)

        # Acceleration
        df_unique = self.time_series_df.drop_duplicates(subset=['company_name', 'days_ago'], keep='last')
        required_days = {0, 30, 60}
        company_day_counts = df_unique.groupby('company_name')['days_ago'].apply(lambda x: set(x))
        complete_companies = company_day_counts[company_day_counts.apply(lambda x: required_days.issubset(x))].index
        df_filtered = df_unique[df_unique['company_name'].isin(complete_companies) & df_unique['days_ago'].isin(required_days)]
        
        if df_filtered.empty:
            print("가속도 분석에 필요한 데이터가 부족합니다 (0, 30, 60일 전 데이터 필요).")
            top_visitor_accel = pd.DataFrame()
            top_blog_accel = pd.DataFrame()
        else:
            df_pivot = df_filtered.pivot(index='company_name', columns='days_ago', values=['visitor_reviews', 'blog_reviews', 'rank'])
            df_pivot.columns = [f"{val}_{days}" for val, days in df_pivot.columns]
            for review_type in ['visitor_reviews', 'blog_reviews']:
                df_pivot[f'acceleration_{review_type}'] = (df_pivot[f'{review_type}_0'] - df_pivot[f'{review_type}_30']) - (df_pivot[f'{review_type}_30'] - df_pivot[f'{review_type}_60'])
            df_pivot['rank_change_60d'] = df_pivot['rank_60'] - df_pivot['rank_0']
            top_visitor_accel = df_pivot.sort_values(by='acceleration_visitor_reviews', ascending=False).head(5).reset_index()
            top_blog_accel = df_pivot.sort_values(by='acceleration_blog_reviews', ascending=False).head(5).reset_index()

        plot_path_visitor_accel = os.path.join(self.result_path, 'v9_visitor_acceleration.png')
        self._plot_acceleration(top_visitor_accel, 'visitor_reviews', plot_path_visitor_accel)
        plot_path_blog_accel = os.path.join(self.result_path, 'v9_blog_acceleration.png')
        self._plot_acceleration(top_blog_accel, 'blog_reviews', plot_path_blog_accel)
        
        data_for_llm = self._md_to_df_string({
            "gatekeepers": gatekeeper_names, 
            "top_visitor_accel": top_visitor_accel,
            "top_blog_accel": top_blog_accel
        })
        
        prompt = f"""## 경쟁 구도 심층 분석 ##
        '동탄치과' 시장의 경쟁 구도를 '수문장 그룹'과 '가속도 그룹'으로 나누어 분석해주세요.
        - '수문장 그룹'의 특징과 이들의 순위 변동이 시장에 미치는 영향을 설명해주세요.
        - '가속도 그룹'이 보여주는 최근 성장세의 특징을 방문자 리뷰와 블로그 리뷰로 나누어 설명해주세요.
        ---
        [수문장 그룹 목록]\n{data_for_llm['gatekeepers']}
        [방문자 리뷰 가속도 TOP 5]\n{data_for_llm['top_visitor_accel']}
        [블로그 리뷰 가속도 TOP 5]\n{data_for_llm['top_blog_accel']}"""
        
        response = self.agent.run(prompt, max_tokens=8000)
        return {
            "title": "경쟁 구도 심층 분석: 수문장과 가속 그룹",
            "content": response.content if response else "",
            "plot_paths": [plot_path_gatekeeper, plot_path_visitor_accel, plot_path_blog_accel]
        }

    def _plot_gatekeepers(self, df, file_path):
        plt.figure(figsize=(12, 8))
        if not df.empty:
            sns.lineplot(data=df, x='days_ago', y='rank', hue='company_name', marker='o', palette='Paired')
            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            plt.title('상위권 수문장 그룹 60일간 순위 변동 추이', fontsize=16, pad=20)
            plt.xlabel('일 전 (Days Ago)')
            plt.ylabel('순위 (Rank)')
            plt.legend(title='병원명', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_acceleration(self, df, review_type, file_path):
        plt.figure(figsize=(12, 7))
        type_kor = '블로그 리뷰' if review_type == 'blog_reviews' else '방문자 리뷰'
        if not df.empty:
            df_sorted = df.sort_values(by=f'acceleration_{review_type}', ascending=False)
            sns.barplot(x='company_name', y=f'acceleration_{review_type}', data=df_sorted, palette='viridis', hue='company_name', legend=False)
            plt.ylabel('리뷰 증가 가속도')
            plt.xlabel('병원명')
            plt.title(f'\'{type_kor}\' 증가 가속도 TOP 5', fontsize=16)
            plt.xticks(rotation=15, ha='right')
            ax2 = plt.gca().twinx()
            sns.lineplot(x='company_name', y='rank_change_60d', data=df_sorted, ax=ax2, color='r', marker='o', label='60일간 순위 변동폭')
            ax2.set_ylabel('60일간 순위 변동 (칸)')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_successful_climbers(self):
        current_data = self.time_series_df[self.time_series_df['days_ago'] == 0].copy()
        past_data = self.time_series_df[self.time_series_df['days_ago'] == 60].copy()
        merged_data = pd.merge(current_data, past_data, on='company_name', suffixes=('_current', '_past'))
        merged_data['rank_improvement'] = merged_data['rank_past'] - merged_data['rank_current']
        merged_data['visitor_review_increase'] = merged_data['visitor_reviews_current'] - merged_data['visitor_reviews_past']
        merged_data['blog_review_increase'] = merged_data['blog_reviews_current'] - merged_data['blog_reviews_past']
        top_climbers = merged_data[merged_data['rank_improvement'] > 0].sort_values(by='rank_improvement', ascending=False).head(10)
        plot_path_climbers = os.path.join(self.result_path, 'v9_top_climbers.png')
        self._plot_top_climbers(top_climbers, plot_path_climbers)
        
        data_for_llm = self._md_to_df_string({"top_climbers": top_climbers})
        prompt = f"""## 성공 방정식 분석 ##
        '도전자' 그룹(순위 급상승 업체)의 성공 요인을 방문자 리뷰와 블로그 리뷰의 역할로 나누어 분석해주세요.
        - 블로그 리뷰가 '가속 엔진' 역할을 한 사례를 구체적인 수치와 함께 설명해주세요.
        - 방문자 리뷰가 '신뢰 기반' 및 '순위 방어선' 역할을 한 사례를 설명해주세요.
        - 최종적으로 '성공 방정식'을 정의해주세요.
        ---
        [TOP 10 순위 상승 업체 상세 데이터]\n{data_for_llm['top_climbers']}"""
        
        response = self.agent.run(prompt, max_tokens=8000)
        return {
            "title": "[핵심] 성공 방정식: '도전자'는 어떻게 순위를 올리는가?",
            "content": response.content if response else "",
            "plot_paths": [plot_path_climbers],
            "data_for_next_step": merged_data
        }

    def _plot_top_climbers(self, climbers_data, file_path):
        top_5_climbers = climbers_data.head(5)
        if top_5_climbers.empty: 
            print("순위 급상승 업체가 없어 해당 그래프를 생성하지 않습니다.")
            return
        fig, axes = plt.subplots(len(top_5_climbers), 1, figsize=(12, 6 * len(top_5_climbers)), sharex=True, squeeze=False)
        axes = axes.flatten()
        fig.suptitle('순위 급상승 TOP 5 업체 심층 분석', fontsize=20, y=1.02)
        for i, (idx, climber) in enumerate(top_5_climbers.iterrows()):
            ax = axes[i]
            company_name = climber['company_name']
            company_ts_data = self.time_series_df[self.time_series_df['company_name'] == company_name].sort_values(by='days_ago', ascending=False)
            if company_ts_data.empty: continue

            ax.plot(company_ts_data['days_ago'], company_ts_data['rank'], marker='o', color='royalblue', label='순위 변동')
            ax.set_ylabel('순위 (Rank)')
            ax.invert_yaxis()
            ax.invert_xaxis()
            ax2 = ax.twinx()
            
            company_ts_data = company_ts_data.sort_values('days_ago', ascending=True)
            company_ts_data['visitor_reviews_increase'] = company_ts_data['visitor_reviews'].diff().fillna(0)
            company_ts_data['blog_reviews_increase'] = company_ts_data['blog_reviews'].diff().fillna(0)
            company_ts_data = company_ts_data.sort_values('days_ago', ascending=False)

            bar_width = (company_ts_data['days_ago'].diff(-1).abs().min() or 5) * 0.4
            ax2.bar(company_ts_data['days_ago'], company_ts_data['visitor_reviews_increase'], color='lightcoral', alpha=0.6, width=bar_width, label='방문자 리뷰 증가량')
            ax2.bar(company_ts_data['days_ago'], company_ts_data['blog_reviews_increase'], bottom=company_ts_data['visitor_reviews_increase'], color='skyblue', alpha=0.6, width=bar_width, label='블로그 리뷰 증가량')
            ax2.set_ylabel('리뷰 증가량')
            title_text = f"[{i+1}위] {company_name}: {int(climber['rank_past'])}위 → {int(climber['rank_current'])}위 ({int(climber['rank_improvement'])}계단 상승)"
            ax.set_title(title_text, fontsize=14)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        axes[-1].set_xlabel('일 전 (Days Ago)')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_target_clinic(self, all_rank_changes):
        target_clinic_info = all_rank_changes[all_rank_changes['company_name'] == TARGET_CLINIC]
        rank_change = 0
        if not target_clinic_info.empty:
            rank_improvement = target_clinic_info.iloc[0]['rank_improvement']
            rank_change = int(rank_improvement * -1) if pd.notna(rank_improvement) else 0

        current_data = self.time_series_df[self.time_series_df['days_ago'] == 0].copy()
        df_with_tiers = self._create_dynamic_tiers_kmeans(current_data, 'visitor_reviews')
        df_with_tiers = self._create_dynamic_tiers_kmeans(df_with_tiers, 'blog_reviews')
        target_clinic_tier_info_df = df_with_tiers[df_with_tiers['company_name'] == TARGET_CLINIC]
        
        tier_info = "정보 없음"
        if not target_clinic_tier_info_df.empty:
            visit_t = target_clinic_tier_info_df.iloc[0].get('visitor_reviews_tier', 'N/A')
            blog_t = target_clinic_tier_info_df.iloc[0].get('blog_reviews_tier', 'N/A')
            tier_info = f"방문자 리뷰 {visit_t}, 블로그 리뷰 {blog_t}"
        
        rank_change_text = f"{abs(rank_change)}계단 {'하락' if rank_change > 0 else '상승'}" if rank_change != 0 else "변동 없음"

        prompt = f"""## '{TARGET_CLINIC}' 맞춤 진단 및 전략 제안 ##
        '내이튼치과의원'의 현재 상황을 진단하고, 미래 성장 전략을 제안해주세요.
        - **60일간 순위가 {rank_change_text}**했다는 사실을 명확히 언급해주세요.
        - 강점(방문자 리뷰)과 약점(블로그 리뷰)을 Tier 데이터와 경쟁사 상황을 근거로 분석해주세요.
        - 최종 결론과 함께 즉시 실행 가능한 구체적인 Action Plan을 제안해주세요.
        ---
        - **현재 Tier:** {tier_info}
        - **60일간 순위 변동:** {rank_change_text}"""
        
        response = self.agent.run(prompt, max_tokens=8000)
        return {
            "title": f"우리의 현주소: '{TARGET_CLINIC}' 종합 진단 및 최종 전략",
            "content": response.content if response else "",
            "plot_paths": []
        }

    def generate_html_report_v9_4(self, report_sections):
        print("\nHTML 보고서 생성 (v9.4)...")
        html_content = ""
        for section in report_sections:
            if not section: continue
            html_content += f"<h2>{section.get('title', 'Untitled')}</h2>\n"
            content_html = self.markdown_converter.convert(section.get('content', ''))
            html_content += content_html
            for plot_path in section.get('plot_paths', []):
                if plot_path and os.path.exists(plot_path):
                    try:
                        with open(plot_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode()
                        html_content += f'<div style="text-align: center; margin: 20px 0;"><img src="data:image/png;base64,{encoded_string}" alt="{os.path.basename(plot_path)}" style="max-width: 90%; height: auto; border: 1px solid #ddd; border-radius: 5px;"></div>\n'
                    except Exception as e:
                        print(f"이미지 파일 '{plot_path}' 처리 중 오류: {e}")
                else:
                    print(f"경고: 플롯 이미지 경로를 찾을 수 없습니다: {plot_path}")
        
        html_template = f"""
        <!DOCTYPE html><html><head><meta charset="UTF-8"><title>'동탄치과' 시장 동적 분석 보고서 v9.4</title>
        <style>body{{font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;line-height:1.6;padding:20px;max-width:1000px;margin:auto;color:#333;}}h1,h2,h3{{color:#2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;}} table{{border-collapse:collapse;width:100%;margin: 20px 0; box-shadow: 0 2px 3px rgba(0,0,0,0.1);}}th,td{{border:1px solid #ddd;padding:12px;text-align:left;}}th{{background-color:#3498db;color:white;}} tr:nth-child(even){{background-color:#f2f9fd;}} img{{max-width:100%;height:auto;display:block;margin:20px auto;border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}}</style>
        </head><body><h1>[V9.4] '동탄치과' 시장 동적 분석 및 경쟁 전략 심화 보고서</h1>{html_content}</body></html>
        """
        
        with open(REPORT_OUTPUT_PATH_HTML, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"HTML 보고서가 다음 경로에 저장되었습니다: {REPORT_OUTPUT_PATH_HTML}")

def main():
    analyzer = RankAnalyzerV9(data_dir=RAW_DATA_DIR, result_path=RESULT_PATH)
    if analyzer.time_series_df.empty:
        print("초기 데이터 로딩에 실패하여 분석을 중단합니다.")
        return

    report_sections = []
    
    # 1. 시장 동향 요약
    report_sections.append(analyzer.analyze_tier_trends())
    
    # 2. 경쟁 구도 심층 분석
    report_sections.append(analyzer.analyze_competitor_landscape())
    
    # 3. 성공 방정식
    climbers_analysis = analyzer.analyze_successful_climbers()
    if climbers_analysis:
        all_rank_changes = climbers_analysis.pop("data_for_next_step", pd.DataFrame())
        report_sections.append(climbers_analysis)
    else:
        all_rank_changes = pd.DataFrame()

    # 4. 우리의 현주소 및 전략
    report_sections.append(analyzer.analyze_target_clinic(all_rank_changes))
    
    analyzer.generate_html_report_v9_4(report_sections)

if __name__ == "__main__":
    main()
