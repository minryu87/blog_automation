import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os

def analyze_cafe_search_correlation():
    """
    카페 조회수와 검색량 간의 상관관계를 분석합니다.
    """
    # 데이터 정의
    data = {
        'month': ['2024-06', '2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12', 
                 '2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06', '2025-07'],
        
        # 내이튼치과
        '내이튼치과_카페': [438, 1531, 2861, 2737, 869, 2217, 2017, 1650, 1222, 1941, 2710, 1626, 1624, 2700],
        '내이튼치과_검색': [1230, 1160, 1310, 1260, 1490, 1260, 1240, 1280, 1420, 1420, 1470, 1390, 1380, 1440],
        
        # 이즈치과
        '이즈치과_카페': [2971, 6672, 5722, 5978, 5145, 10485, 4774, 3633, 2071, 5908, 5181, 1947, 1244, 3467],
        '이즈치과_검색': [1920, 2060, 1600, 1550, 1740, 1630, 1850, 1910, 2160, 2090, 1910, 1710, 1510, 1690],
        
        # 동탄연세바로치과
        '동탄연세바로치과_카페': [2241, 10086, 4332, 1187, 2773, 5634, 4301, 5290, 4716, 7529, 5364, 4005, 2916, 4474],
        '동탄연세바로치과_검색': [760, 940, 920, 890, 870, 790, 1060, 1250, 1270, 1290, 1350, 1000, 900, 1220],
        
        # 플란치과의원 경기 동탄점
        '플란치과의원 경기 동탄점_카페': [971, 1367, 0, 288, 1459, 1116, 3333, 5285, 3487, 3001, 2065, 1609, 4144, 7374],
        '플란치과의원 경기 동탄점_검색': [320, 410, 330, 230, 250, 240, 240, 250, 200, 270, 190, 250, 180, 3590]
    }
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 출력 디렉토리 생성
    output_dir = 'cafe_search_correlation_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 카페 조회수와 검색량 상관관계 분석 ===")
    
    # 1. 기본 통계 분석
    print("\n=== 기본 통계 ===")
    hospitals = ['내이튼치과', '이즈치과', '동탄연세바로치과', '플란치과의원 경기 동탄점']
    
    for hospital in hospitals:
        cafe_col = f'{hospital}_카페'
        search_col = f'{hospital}_검색'
        
        if cafe_col in df.columns and search_col in df.columns:
            cafe_data = df[cafe_col]
            search_data = df[search_col]
            
            # 상관계수 계산
            correlation = cafe_data.corr(search_data)
            spearman_corr = cafe_data.corr(search_data, method='spearman')
            
            print(f"\n{hospital}:")
            print(f"  카페 조회수 평균: {cafe_data.mean():.0f}")
            print(f"  검색량 평균: {search_data.mean():.0f}")
            print(f"  피어슨 상관계수: {correlation:.3f}")
            print(f"  스피어만 상관계수: {spearman_corr:.3f}")
    
    # 2. 시계열 그래프 생성
    create_time_series_plots(df, hospitals, output_dir)
    
    # 3. 산점도 및 상관관계 그래프
    create_correlation_plots(df, hospitals, output_dir)
    
    # 4. 증감 분석
    analyze_changes(df, hospitals, output_dir)
    
    # 5. 지연 효과 분석
    analyze_lag_effects(df, hospitals, output_dir)
    
    # 6. CSV 저장
    save_analysis_results(df, hospitals, output_dir)
    
    return df

def create_time_series_plots(df, hospitals, output_dir):
    """시계열 그래프를 생성합니다."""
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, hospital in enumerate(hospitals):
        cafe_col = f'{hospital}_카페'
        search_col = f'{hospital}_검색'
        
        if cafe_col in df.columns and search_col in df.columns:
            ax = axes[i]
            
            # 이중 Y축 그래프
            ax2 = ax.twinx()
            
            # 카페 조회수 (왼쪽 Y축)
            line1 = ax.plot(df['month'], df[cafe_col], 'b-o', label='카페 조회수', linewidth=2)
            ax.set_ylabel('카페 조회수', color='b', fontsize=10)
            ax.tick_params(axis='y', labelcolor='b')
            
            # 검색량 (오른쪽 Y축)
            line2 = ax2.plot(df['month'], df[search_col], 'r-s', label='검색량', linewidth=2)
            ax2.set_ylabel('검색량', color='r', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='r')
            
            ax.set_title(f'{hospital} - 카페 조회수 vs 검색량', fontsize=12, fontweight='bold')
            ax.set_xlabel('월', fontsize=10)
            
            # X축 레이블 회전
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 범례
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'time_series_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n시계열 비교 그래프 저장: {output_file}")
    plt.close()

def create_correlation_plots(df, hospitals, output_dir):
    """상관관계 산점도를 생성합니다."""
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, hospital in enumerate(hospitals):
        cafe_col = f'{hospital}_카페'
        search_col = f'{hospital}_검색'
        
        if cafe_col in df.columns and search_col in df.columns:
            ax = axes[i]
            
            # 산점도
            ax.scatter(df[cafe_col], df[search_col], alpha=0.7, s=100)
            
            # 회귀선
            z = np.polyfit(df[cafe_col], df[search_col], 1)
            p = np.poly1d(z)
            ax.plot(df[cafe_col], p(df[cafe_col]), "r--", alpha=0.8)
            
            # 상관계수 표시
            correlation = df[cafe_col].corr(df[search_col])
            ax.text(0.05, 0.95, f'상관계수: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('카페 조회수', fontsize=10)
            ax.set_ylabel('검색량', fontsize=10)
            ax.set_title(f'{hospital} - 상관관계 분석', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'correlation_scatter_plots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"상관관계 산점도 저장: {output_file}")
    plt.close()

def analyze_changes(df, hospitals, output_dir):
    """증감 분석을 수행합니다."""
    print("\n=== 증감 분석 ===")
    
    changes_data = []
    
    for hospital in hospitals:
        cafe_col = f'{hospital}_카페'
        search_col = f'{hospital}_검색'
        
        if cafe_col in df.columns and search_col in df.columns:
            # 월별 증감 계산
            cafe_changes = df[cafe_col].diff()
            search_changes = df[search_col].diff()
            
            # 증감 상관관계
            change_correlation = cafe_changes.corr(search_changes)
            
            # 증감 일치율 (같은 방향으로 변화하는 비율)
            same_direction = ((cafe_changes > 0) & (search_changes > 0)) | ((cafe_changes < 0) & (search_changes < 0))
            match_rate = same_direction.sum() / len(cafe_changes.dropna())
            
            changes_data.append({
                'hospital': hospital,
                'change_correlation': change_correlation,
                'match_rate': match_rate,
                'avg_cafe_change': cafe_changes.mean(),
                'avg_search_change': search_changes.mean()
            })
            
            print(f"\n{hospital}:")
            print(f"  증감 상관계수: {change_correlation:.3f}")
            print(f"  증감 일치율: {match_rate:.1%}")
            print(f"  평균 카페 증감: {cafe_changes.mean():.0f}")
            print(f"  평균 검색 증감: {search_changes.mean():.0f}")
    
    # 증감 분석 결과 저장
    changes_df = pd.DataFrame(changes_data)
    changes_file = os.path.join(output_dir, 'changes_analysis.csv')
    changes_df.to_csv(changes_file, index=False, encoding='utf-8-sig')
    print(f"\n증감 분석 결과 저장: {changes_file}")

def analyze_lag_effects(df, hospitals, output_dir):
    """지연 효과를 분석합니다."""
    print("\n=== 지연 효과 분석 ===")
    
    lag_data = []
    
    for hospital in hospitals:
        cafe_col = f'{hospital}_카페'
        search_col = f'{hospital}_검색'
        
        if cafe_col in df.columns and search_col in df.columns:
            cafe_data = df[cafe_col]
            search_data = df[search_col]
            
            # 1개월, 2개월 지연 상관관계 계산
            correlations = []
            for lag in range(1, 4):  # 1~3개월 지연
                if len(cafe_data) > lag:
                    lagged_cafe = cafe_data.shift(lag).dropna()
                    current_search = search_data.loc[lagged_cafe.index]
                    if len(lagged_cafe) == len(current_search):
                        corr = lagged_cafe.corr(current_search)
                        correlations.append(corr)
                    else:
                        correlations.append(np.nan)
                else:
                    correlations.append(np.nan)
            
            lag_data.append({
                'hospital': hospital,
                'lag_1_month': correlations[0] if len(correlations) > 0 else np.nan,
                'lag_2_months': correlations[1] if len(correlations) > 1 else np.nan,
                'lag_3_months': correlations[2] if len(correlations) > 2 else np.nan
            })
            
            print(f"\n{hospital} 지연 효과:")
            for i, corr in enumerate(correlations, 1):
                if not np.isnan(corr):
                    print(f"  {i}개월 지연 상관계수: {corr:.3f}")
    
    # 지연 효과 분석 결과 저장
    lag_df = pd.DataFrame(lag_data)
    lag_file = os.path.join(output_dir, 'lag_effects_analysis.csv')
    lag_df.to_csv(lag_file, index=False, encoding='utf-8-sig')
    print(f"\n지연 효과 분석 결과 저장: {lag_file}")

def save_analysis_results(df, hospitals, output_dir):
    """분석 결과를 CSV로 저장합니다."""
    # 전체 데이터 저장
    output_file = os.path.join(output_dir, 'cafe_search_data.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n전체 데이터 저장: {output_file}")
    
    # 상관관계 요약
    summary_data = []
    for hospital in hospitals:
        cafe_col = f'{hospital}_카페'
        search_col = f'{hospital}_검색'
        
        if cafe_col in df.columns and search_col in df.columns:
            correlation = df[cafe_col].corr(df[search_col])
            spearman_corr = df[cafe_col].corr(df[search_col], method='spearman')
            
            summary_data.append({
                'hospital': hospital,
                'pearson_correlation': correlation,
                'spearman_correlation': spearman_corr,
                'avg_cafe_views': df[cafe_col].mean(),
                'avg_search_volume': df[search_col].mean(),
                'cafe_std': df[cafe_col].std(),
                'search_std': df[search_col].std()
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, 'correlation_summary.csv')
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"상관관계 요약 저장: {summary_file}")

if __name__ == "__main__":
    analyze_cafe_search_correlation()
