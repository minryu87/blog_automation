import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import re
import matplotlib.font_manager as fm
import platform
import os
warnings.filterwarnings('ignore')

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    
    if os.path.exists(font_path):
        try:
            # 폰트 추가
            fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            
            # matplotlib 설정
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            
            print(f"한글 폰트 설정 완료: {font_name}")
            return True
        except Exception as e:
            print(f"폰트 설정 실패: {e}")
    
    # 폰트를 찾지 못한 경우 기본 설정
    print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return False

# 한글 폰트 설정
setup_korean_font()

# 데이터 로드 및 전처리
df = pd.read_csv('data/raw/booking_record_detail.csv')

# 날짜 파싱 함수
def parse_datetime(date_str):
    if pd.isna(date_str):
        return pd.NaT
    
    patterns = [
        r'(\d{4}-\d{2}-\d{2}) \(([월화수목금토일])\) 오전 (\d{1,2}):(\d{2}):(\d{2})',
        r'(\d{4}-\d{2}-\d{2}) \(([월화수목금토일])\) 오후 (\d{1,2}):(\d{2}):(\d{2})'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, str(date_str))
        if match:
            date_part, weekday, hour, minute, second = match.groups()
            hour = int(hour)
            
            if '오후' in str(date_str) and hour != 12:
                hour += 12
            elif '오전' in str(date_str) and hour == 12:
                hour = 0
            
            hour = hour % 24
            return pd.to_datetime(f"{date_part} {hour:02d}:{minute}:{second}")
    
    return pd.NaT

df['예약신청일시'] = df['예약신청일시'].apply(parse_datetime)

# 치료과목 및 예약경로 데이터 추출
treatment_columns = ['치료과목1', '치료과목2', '치료과목3', '치료과목4', '치료과목5', '치료과목6']
booking_columns = ['예약 경로1', '예약 경로2', '예약 경로3', '예약 경로4', '예약 경로5']

def create_korean_analysis():
    """한글 레이블을 사용한 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🏥 병원 예약 데이터 분석', fontsize=16, fontweight='bold')
    
    # 1. 유입경로별 예약 건수
    inflow_counts = df['유입경로'].value_counts()
    axes[0, 0].pie(inflow_counts.values, labels=inflow_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('유입경로별 예약 비율')
    
    # 2. 치료과목별 예약 건수
    all_treatments = []
    for col in treatment_columns:
        treatments = df[col].dropna()
        all_treatments.extend(treatments.tolist())
    
    treatment_counts = Counter(all_treatments)
    treatments = list(treatment_counts.keys())
    counts = list(treatment_counts.values())
    
    axes[0, 1].barh(treatments, counts, color='lightcoral')
    axes[0, 1].set_title('치료과목별 예약 건수')
    axes[0, 1].set_xlabel('예약 건수')
    
    # 3. 신환 비율
    new_patient_rate = (df['신환여부'] == 'Y').sum() / len(df) * 100
    
    categories = ['신환', '재방문']
    rates = [new_patient_rate, 100 - new_patient_rate]
    colors = ['lightblue', 'lightgreen']
    
    bars = axes[1, 0].bar(categories, rates, color=colors)
    axes[1, 0].set_title('환자 유형 분포')
    axes[1, 0].set_ylabel('비율 (%)')
    
    # 값 표시
    for bar, rate in zip(bars, rates):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{rate:.1f}%', ha='center', va='bottom')
    
    # 4. 월별 예약 트렌드
    df['월'] = df['예약신청일시'].dt.month
    monthly_bookings = df.groupby('월').size()
    
    axes[1, 1].plot(monthly_bookings.index, monthly_bookings.values, marker='o', linewidth=2, color='purple')
    axes[1, 1].set_title('월별 예약 트렌드')
    axes[1, 1].set_xlabel('월')
    axes[1, 1].set_ylabel('예약 건수')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/analysis_korean.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_korean_analysis():
    """한글 레이블을 사용한 상세 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🏥 병원 예약 상세 분석', fontsize=16, fontweight='bold')
    
    # 1. 치료과목별 신환 비율
    treatment_new_patient = {}
    for idx, row in df.iterrows():
        treatments = [row[col] for col in treatment_columns if pd.notna(row[col]) and row[col]]
        is_new = row['신환여부'] == 'Y'
        
        for treatment in treatments:
            if treatment not in treatment_new_patient:
                treatment_new_patient[treatment] = {'total': 0, 'new': 0}
            treatment_new_patient[treatment]['total'] += 1
            if is_new:
                treatment_new_patient[treatment]['new'] += 1
    
    treatment_names = list(treatment_new_patient.keys())
    new_rates = [treatment_new_patient[t]['new'] / treatment_new_patient[t]['total'] * 100 for t in treatment_names]
    
    bars = axes[0, 0].bar(treatment_names, new_rates, color='lightblue')
    axes[0, 0].set_title('치료과목별 신환 비율')
    axes[0, 0].set_ylabel('신환 비율 (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 값 표시
    for bar, rate in zip(bars, new_rates):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{rate:.1f}%', ha='center', va='bottom')
    
    # 2. 예약경로 분석
    all_booking_paths = []
    for col in booking_columns:
        paths = df[col].dropna()
        all_booking_paths.extend(paths.tolist())
    
    booking_counts = Counter(all_booking_paths)
    top_booking_paths = booking_counts.most_common(8)
    paths = [path for path, _ in top_booking_paths]
    counts = [count for _, count in top_booking_paths]
    
    axes[0, 1].barh(paths, counts, color='lightgreen')
    axes[0, 1].set_title('주요 예약경로')
    axes[0, 1].set_xlabel('예약 건수')
    
    # 3. 요일별 예약 패턴
    df['요일'] = df['예약신청일시'].dt.day_name()
    weekday_counts = df['요일'].value_counts()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = weekday_counts.reindex(weekday_order)
    
    # 요일을 한글로 변환
    weekday_korean = {
        'Monday': '월요일',
        'Tuesday': '화요일', 
        'Wednesday': '수요일',
        'Thursday': '목요일',
        'Friday': '금요일',
        'Saturday': '토요일',
        'Sunday': '일요일'
    }
    
    weekday_labels = [weekday_korean.get(day, day) for day in weekday_counts.index]
    
    bars = axes[1, 0].bar(weekday_labels, weekday_counts.values, color='lightcoral')
    axes[1, 0].set_title('요일별 예약 패턴')
    axes[1, 0].set_ylabel('예약 건수')
    
    # 값 표시
    for bar, count in zip(bars, weekday_counts.values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                       str(count), ha='center', va='bottom')
    
    # 4. 시간대별 예약 패턴
    df['시간'] = df['예약신청일시'].dt.hour
    hourly_bookings = df.groupby('시간').size()
    
    axes[1, 1].plot(hourly_bookings.index, hourly_bookings.values, marker='o', linewidth=2, color='orange')
    axes[1, 1].set_title('시간대별 예약 패턴')
    axes[1, 1].set_xlabel('시간')
    axes[1, 1].set_ylabel('예약 건수')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/detailed_analysis_korean.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_treatment_inflow_analysis():
    """치료과목별 유입경로 분석"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🦷 치료과목별 유입경로 분석', fontsize=16, fontweight='bold')
    
    # 치료과목별 유입경로 데이터 수집
    treatment_inflow = {}
    for idx, row in df.iterrows():
        treatments = [row[col] for col in treatment_columns if pd.notna(row[col]) and row[col]]
        inflow = row['유입경로']
        
        for treatment in treatments:
            if treatment not in treatment_inflow:
                treatment_inflow[treatment] = Counter()
            treatment_inflow[treatment][inflow] += 1
    
    # 상위 4개 치료과목 선택
    all_treatments = []
    for col in treatment_columns:
        treatments = df[col].dropna()
        all_treatments.extend(treatments.tolist())
    
    treatment_counts = Counter(all_treatments)
    top_treatments = [t for t, _ in treatment_counts.most_common(4)]
    
    for i, treatment in enumerate(top_treatments):
        row, col = i // 2, i % 2
        inflows = treatment_inflow[treatment]
        
        # 상위 5개 유입경로만 선택
        top_inflows = inflows.most_common(5)
        labels = [label for label, _ in top_inflows]
        values = [value for _, value in top_inflows]
        
        axes[row, col].pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[row, col].set_title(f'{treatment} 유입경로')
    
    plt.tight_layout()
    plt.savefig('data/processed/treatment_inflow_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 메인 실행
if __name__ == "__main__":
    print("=== 병원 예약 데이터 분석 (한글 버전) ===\n")
    
    # 결과 디렉토리 생성
    os.makedirs('data/processed', exist_ok=True)
    
    print("1. 기본 분석 시각화 생성 중...")
    create_korean_analysis()
    
    print("2. 상세 분석 시각화 생성 중...")
    create_detailed_korean_analysis()
    
    print("3. 치료과목별 유입경로 분석 생성 중...")
    create_treatment_inflow_analysis()
    
    print("\n=== 분석 완료 ===")
    print("생성된 파일:")
    print("- data/processed/analysis_korean.png")
    print("- data/processed/detailed_analysis_korean.png")
    print("- data/processed/treatment_inflow_analysis.png")
    
    # 주요 통계 출력
    print("\n=== 주요 통계 ===")
    print(f"총 예약 건수: {len(df):,}건")
    print(f"신환 비율: {(df['신환여부'] == 'Y').sum() / len(df) * 100:.1f}%")
    print(f"이용완료율: {(df['상태'] == '이용완료').sum() / len(df) * 100:.1f}%")
    
    # 상위 유입경로
    top_inflow = df['유입경로'].value_counts().head(3)
    print("\n상위 3개 유입경로:")
    for i, (path, count) in enumerate(top_inflow.items(), 1):
        print(f"  {i}. {path}: {count:,}건 ({count/len(df)*100:.1f}%)")
    
    # 상위 치료과목
    all_treatments = []
    for col in treatment_columns:
        treatments = df[col].dropna()
        all_treatments.extend(treatments.tolist())
    
    treatment_counts = Counter(all_treatments)
    print("\n상위 3개 치료과목:")
    for i, (treatment, count) in enumerate(treatment_counts.most_common(3), 1):
        print(f"  {i}. {treatment}: {count:,}건")
    
    print("\n=== 데이터 특성 안내 ===")
    print("⚠️ 치료과목과 예약경로는 중복 선택이 가능합니다:")
    print("  - 치료과목: 한 예약에 최대 6개까지 선택 가능")
    print("  - 예약경로: 한 예약에 최대 5개까지 선택 가능")
    print("  - 따라서 총 예약 건수 대비 치료과목/예약경로 건수가 더 많을 수 있습니다.")
