import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import re
import os
import platform
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

# 한글 폰트 설정
def find_korean_fonts():
    korean_fonts = []
    if platform.system() == 'Darwin':
        font_paths = ['/System/Library/Fonts/Supplemental/AppleGothic.ttf']
        for font_path in font_paths:
            if os.path.exists(font_path):
                korean_fonts.append(font_path)
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
    print("경고: 적절한 한글 폰트를 찾을 수 없습니다. 그래프의 한글이 깨질 수 있습니다.")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

# 폰트 설정
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
    print("경고: 적절한 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    # 기본 폰트 설정
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'AppleGothic', 'Malgun Gothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-v0_8')

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

# 시각화 함수들
def create_inflow_analysis():
    """유입경로 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🔍 유입경로 분석', fontsize=16, fontweight='bold')
    
    # 1. 유입경로별 예약 건수
    inflow_counts = df['유입경로'].value_counts()
    axes[0, 0].pie(inflow_counts.values, labels=inflow_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('유입경로별 예약 비율')
    
    # 2. 유입경로별 신환 비율
    inflow_new_rate = df.groupby('유입경로')['신환여부'].apply(lambda x: (x == 'Y').sum() / len(x) * 100)
    inflow_new_rate.plot(kind='bar', ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_title('유입경로별 신환 비율 (%)')
    axes[0, 1].set_ylabel('신환 비율 (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 유입경로별 이용완료율
    inflow_completion_rate = df.groupby('유입경로')['상태'].apply(lambda x: (x == '이용완료').sum() / len(x) * 100)
    inflow_completion_rate.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('유입경로별 이용완료율 (%)')
    axes[1, 0].set_ylabel('이용완료율 (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 월별 유입경로 트렌드
    df['월'] = df['예약신청일시'].dt.month
    monthly_inflow = df.groupby(['월', '유입경로']).size().unstack(fill_value=0)
    monthly_inflow.plot(kind='line', ax=axes[1, 1], marker='o')
    axes[1, 1].set_title('월별 유입경로 트렌드')
    axes[1, 1].set_xlabel('월')
    axes[1, 1].set_ylabel('예약 건수')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('data/processed/inflow_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_treatment_analysis():
    """치료과목 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🦷 치료과목 분석', fontsize=16, fontweight='bold')
    
    # 치료과목 데이터 추출
    all_treatments = []
    for col in treatment_columns:
        treatments = df[col].dropna()
        all_treatments.extend(treatments.tolist())
    
    treatment_counts = Counter(all_treatments)
    
    # 1. 치료과목별 예약 건수
    treatments = list(treatment_counts.keys())
    counts = list(treatment_counts.values())
    axes[0, 0].barh(treatments, counts, color='lightcoral')
    axes[0, 0].set_title('치료과목별 예약 건수')
    axes[0, 0].set_xlabel('예약 건수')
    
    # 2. 치료과목별 신환 비율
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
    
    bars = axes[0, 1].bar(treatment_names, new_rates, color='lightblue')
    axes[0, 1].set_title('치료과목별 신환 비율 (%)')
    axes[0, 1].set_ylabel('신환 비율 (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 값 표시
    for bar, rate in zip(bars, new_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{rate:.1f}%', ha='center', va='bottom')
    
    # 3. 치료과목별 유입경로 히트맵
    treatment_inflow = {}
    for idx, row in df.iterrows():
        treatments = [row[col] for col in treatment_columns if pd.notna(row[col]) and row[col]]
        inflow = row['유입경로']
        
        for treatment in treatments:
            if treatment not in treatment_inflow:
                treatment_inflow[treatment] = Counter()
            treatment_inflow[treatment][inflow] += 1
    
    # 히트맵 데이터 준비
    inflow_paths = df['유입경로'].unique()
    heatmap_data = []
    for treatment in treatment_names:
        row = []
        for inflow in inflow_paths:
            row.append(treatment_inflow.get(treatment, Counter()).get(inflow, 0))
        heatmap_data.append(row)
    
    im = axes[1, 0].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_title('치료과목별 유입경로 히트맵')
    axes[1, 0].set_xticks(range(len(inflow_paths)))
    axes[1, 0].set_xticklabels(inflow_paths, rotation=45, ha='right')
    axes[1, 0].set_yticks(range(len(treatment_names)))
    axes[1, 0].set_yticklabels(treatment_names)
    
    # 컬러바 추가
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. 월별 치료과목 트렌드
    monthly_treatment = {}
    for idx, row in df.iterrows():
        month = row['예약신청일시'].month
        treatments = [row[col] for col in treatment_columns if pd.notna(row[col]) and row[col]]
        
        if month not in monthly_treatment:
            monthly_treatment[month] = Counter()
        for treatment in treatments:
            monthly_treatment[month][treatment] += 1
    
    # 상위 5개 치료과목만 선택
    top_treatments = [t for t, _ in treatment_counts.most_common(5)]
    
    for treatment in top_treatments:
        months = sorted(monthly_treatment.keys())
        counts = [monthly_treatment[month].get(treatment, 0) for month in months]
        axes[1, 1].plot(months, counts, marker='o', label=treatment)
    
    axes[1, 1].set_title('월별 치료과목 트렌드 (상위 5개)')
    axes[1, 1].set_xlabel('월')
    axes[1, 1].set_ylabel('예약 건수')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('data/processed/treatment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_booking_path_analysis():
    """예약경로 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📞 예약경로 분석', fontsize=16, fontweight='bold')
    
    # 예약경로 데이터 추출
    all_booking_paths = []
    for col in booking_columns:
        paths = df[col].dropna()
        all_booking_paths.extend(paths.tolist())
    
    booking_counts = Counter(all_booking_paths)
    
    # 1. 예약경로별 건수 (상위 10개)
    top_booking_paths = booking_counts.most_common(10)
    paths = [path for path, _ in top_booking_paths]
    counts = [count for _, count in top_booking_paths]
    
    axes[0, 0].barh(paths, counts, color='lightgreen')
    axes[0, 0].set_title('예약경로별 건수 (상위 10개)')
    axes[0, 0].set_xlabel('예약 건수')
    
    # 2. 예약경로별 신환 비율
    booking_new_rate = {}
    for idx, row in df.iterrows():
        booking_paths = [row[col] for col in booking_columns if pd.notna(row[col]) and row[col]]
        is_new = row['신환여부'] == 'Y'
        
        for path in booking_paths:
            if path not in booking_new_rate:
                booking_new_rate[path] = {'total': 0, 'new': 0}
            booking_new_rate[path]['total'] += 1
            if is_new:
                booking_new_rate[path]['new'] += 1
    
    # 상위 10개 예약경로만 선택
    top_paths = [path for path, _ in top_booking_paths]
    new_rates = [booking_new_rate[path]['new'] / booking_new_rate[path]['total'] * 100 for path in top_paths]
    
    bars = axes[0, 1].bar(top_paths, new_rates, color='lightblue')
    axes[0, 1].set_title('예약경로별 신환 비율 (%)')
    axes[0, 1].set_ylabel('신환 비율 (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 값 표시
    for bar, rate in zip(bars, new_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{rate:.1f}%', ha='center', va='bottom')
    
    # 3. 요일별 예약경로 분석
    df['요일'] = df['예약신청일시'].dt.day_name()
    weekday_booking = {}
    for idx, row in df.iterrows():
        weekday = row['요일']
        booking_paths = [row[col] for col in booking_columns if pd.notna(row[col]) and row[col]]
        
        if weekday not in weekday_booking:
            weekday_booking[weekday] = Counter()
        for path in booking_paths:
            weekday_booking[weekday][path] += 1
    
    # 상위 5개 예약경로만 선택
    top_5_paths = [path for path, _ in top_booking_paths[:5]]
    weekday_data = []
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for weekday in weekdays:
        row = [weekday_booking[weekday].get(path, 0) for path in top_5_paths]
        weekday_data.append(row)
    
    im = axes[1, 0].imshow(weekday_data, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('요일별 예약경로 히트맵')
    axes[1, 0].set_xticks(range(len(top_5_paths)))
    axes[1, 0].set_xticklabels(top_5_paths, rotation=45, ha='right')
    axes[1, 0].set_yticks(range(len(weekdays)))
    axes[1, 0].set_yticklabels(weekdays)
    
    plt.colorbar(im, ax=axes[1, 0])
    
    # 4. 월별 예약경로 트렌드
    monthly_booking = {}
    for idx, row in df.iterrows():
        month = row['예약신청일시'].month
        booking_paths = [row[col] for col in booking_columns if pd.notna(row[col]) and row[col]]
        
        if month not in monthly_booking:
            monthly_booking[month] = Counter()
        for path in booking_paths:
            monthly_booking[month][path] += 1
    
    for path in top_5_paths:
        months = sorted(monthly_booking.keys())
        counts = [monthly_booking[month].get(path, 0) for month in months]
        axes[1, 1].plot(months, counts, marker='o', label=path)
    
    axes[1, 1].set_title('월별 예약경로 트렌드 (상위 5개)')
    axes[1, 1].set_xlabel('월')
    axes[1, 1].set_ylabel('예약 건수')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('data/processed/booking_path_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_temporal_analysis():
    """시계열 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📅 시계열 분석', fontsize=16, fontweight='bold')
    
    # 1. 일별 예약 건수
    daily_bookings = df.groupby(df['예약신청일시'].dt.date).size()
    axes[0, 0].plot(daily_bookings.index, daily_bookings.values, linewidth=1, alpha=0.7)
    axes[0, 0].set_title('일별 예약 건수')
    axes[0, 0].set_xlabel('날짜')
    axes[0, 0].set_ylabel('예약 건수')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 요일별 예약 건수
    df['요일'] = df['예약신청일시'].dt.day_name()
    weekday_counts = df['요일'].value_counts()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = weekday_counts.reindex(weekday_order)
    
    bars = axes[0, 1].bar(weekday_counts.index, weekday_counts.values, color='lightcoral')
    axes[0, 1].set_title('요일별 예약 건수')
    axes[0, 1].set_ylabel('예약 건수')
    
    # 값 표시
    for bar, count in zip(bars, weekday_counts.values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                       str(count), ha='center', va='bottom')
    
    # 3. 월별 예약 건수
    df['월'] = df['예약신청일시'].dt.month
    monthly_bookings = df.groupby('월').size()
    
    bars = axes[1, 0].bar(monthly_bookings.index, monthly_bookings.values, color='lightblue')
    axes[1, 0].set_title('월별 예약 건수')
    axes[1, 0].set_xlabel('월')
    axes[1, 0].set_ylabel('예약 건수')
    
    # 값 표시
    for bar, count in zip(bars, monthly_bookings.values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                       str(count), ha='center', va='bottom')
    
    # 4. 시간대별 예약 건수
    df['시간'] = df['예약신청일시'].dt.hour
    hourly_bookings = df.groupby('시간').size()
    
    axes[1, 1].plot(hourly_bookings.index, hourly_bookings.values, marker='o', linewidth=2)
    axes[1, 1].set_title('시간대별 예약 건수')
    axes[1, 1].set_xlabel('시간')
    axes[1, 1].set_ylabel('예약 건수')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 메인 실행
if __name__ == "__main__":
    print("=== 병원 예약 데이터 시각화 대시보드 ===\n")
    
    # 결과 디렉토리 생성
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    print("1. 유입경로 분석 시각화 생성 중...")
    create_inflow_analysis()
    
    print("2. 치료과목 분석 시각화 생성 중...")
    create_treatment_analysis()
    
    print("3. 예약경로 분석 시각화 생성 중...")
    create_booking_path_analysis()
    
    print("4. 시계열 분석 시각화 생성 중...")
    create_temporal_analysis()
    
    print("\n=== 모든 시각화 완료 ===")
    print("생성된 파일:")
    print("- data/processed/inflow_analysis.png")
    print("- data/processed/treatment_analysis.png")
    print("- data/processed/booking_path_analysis.png")
    print("- data/processed/temporal_analysis.png")
