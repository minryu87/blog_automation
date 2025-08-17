import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import re
warnings.filterwarnings('ignore')

# 기본 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

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

def create_simple_analysis():
    """간단한 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hospital Booking Analysis', fontsize=16, fontweight='bold')
    
    # 1. 유입경로별 예약 건수
    inflow_counts = df['유입경로'].value_counts()
    axes[0, 0].pie(inflow_counts.values, labels=inflow_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Inflow Path Distribution')
    
    # 2. 치료과목별 예약 건수
    all_treatments = []
    for col in treatment_columns:
        treatments = df[col].dropna()
        all_treatments.extend(treatments.tolist())
    
    treatment_counts = Counter(all_treatments)
    treatments = list(treatment_counts.keys())
    counts = list(treatment_counts.values())
    
    axes[0, 1].barh(treatments, counts, color='lightcoral')
    axes[0, 1].set_title('Treatment Categories')
    axes[0, 1].set_xlabel('Number of Bookings')
    
    # 3. 신환 비율
    new_patient_rate = (df['신환여부'] == 'Y').sum() / len(df) * 100
    completion_rate = (df['상태'] == '이용완료').sum() / len(df) * 100
    
    categories = ['New Patients', 'Returning Patients']
    rates = [new_patient_rate, 100 - new_patient_rate]
    colors = ['lightblue', 'lightgreen']
    
    bars = axes[1, 0].bar(categories, rates, color=colors)
    axes[1, 0].set_title('Patient Type Distribution')
    axes[1, 0].set_ylabel('Percentage (%)')
    
    # 값 표시
    for bar, rate in zip(bars, rates):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{rate:.1f}%', ha='center', va='bottom')
    
    # 4. 월별 예약 트렌드
    df['월'] = df['예약신청일시'].dt.month
    monthly_bookings = df.groupby('월').size()
    
    axes[1, 1].plot(monthly_bookings.index, monthly_bookings.values, marker='o', linewidth=2, color='purple')
    axes[1, 1].set_title('Monthly Booking Trend')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Bookings')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/simple_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_treatment_analysis():
    """치료과목 분석"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Treatment Analysis', fontsize=16, fontweight='bold')
    
    # 치료과목 데이터 추출
    all_treatments = []
    for col in treatment_columns:
        treatments = df[col].dropna()
        all_treatments.extend(treatments.tolist())
    
    treatment_counts = Counter(all_treatments)
    
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
    axes[0, 0].set_title('New Patient Rate by Treatment')
    axes[0, 0].set_ylabel('New Patient Rate (%)')
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
    axes[0, 1].set_title('Top Booking Paths')
    axes[0, 1].set_xlabel('Number of Bookings')
    
    # 3. 요일별 예약 패턴
    df['요일'] = df['예약신청일시'].dt.day_name()
    weekday_counts = df['요일'].value_counts()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = weekday_counts.reindex(weekday_order)
    
    bars = axes[1, 0].bar(weekday_counts.index, weekday_counts.values, color='lightcoral')
    axes[1, 0].set_title('Bookings by Day of Week')
    axes[1, 0].set_ylabel('Number of Bookings')
    
    # 값 표시
    for bar, count in zip(bars, weekday_counts.values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                       str(count), ha='center', va='bottom')
    
    # 4. 시간대별 예약 패턴
    df['시간'] = df['예약신청일시'].dt.hour
    hourly_bookings = df.groupby('시간').size()
    
    axes[1, 1].plot(hourly_bookings.index, hourly_bookings.values, marker='o', linewidth=2, color='orange')
    axes[1, 1].set_title('Bookings by Hour of Day')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Number of Bookings')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/treatment_analysis_simple.png', dpi=300, bbox_inches='tight')
    plt.close()

# 메인 실행
if __name__ == "__main__":
    print("=== Hospital Booking Data Analysis (Simple Version) ===\n")
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    print("1. Creating simple analysis visualization...")
    create_simple_analysis()
    
    print("2. Creating treatment analysis visualization...")
    create_treatment_analysis()
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- data/processed/simple_analysis.png")
    print("- data/processed/treatment_analysis_simple.png")
    
    # 주요 통계 출력
    print("\n=== Key Statistics ===")
    print(f"Total bookings: {len(df):,}")
    print(f"New patient rate: {(df['신환여부'] == 'Y').sum() / len(df) * 100:.1f}%")
    print(f"Completion rate: {(df['상태'] == '이용완료').sum() / len(df) * 100:.1f}%")
    
    # 상위 유입경로
    top_inflow = df['유입경로'].value_counts().head(3)
    print("\nTop 3 inflow paths:")
    for i, (path, count) in enumerate(top_inflow.items(), 1):
        print(f"  {i}. {path}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # 상위 치료과목
    all_treatments = []
    for col in treatment_columns:
        treatments = df[col].dropna()
        all_treatments.extend(treatments.tolist())
    
    treatment_counts = Counter(all_treatments)
    print("\nTop 3 treatment categories:")
    for i, (treatment, count) in enumerate(treatment_counts.most_common(3), 1):
        print(f"  {i}. {treatment}: {count:,}")
