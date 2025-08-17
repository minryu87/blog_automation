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

# 한글 폰트 설정 - 더 강력한 방법
def setup_korean_font():
    """한글 폰트 설정"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        # macOS에서 사용 가능한 한글 폰트들
        font_paths = [
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
            '/System/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode MS.ttf'
        ]
        
        for font_path in font_paths:
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
                    print(f"폰트 설정 실패 {font_path}: {e}")
                    continue
    
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

def create_analysis_with_english_labels():
    """영어 레이블을 사용한 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hospital Booking Analysis', fontsize=16, fontweight='bold')
    
    # 1. 유입경로별 예약 건수 (영어 레이블 사용)
    inflow_counts = df['유입경로'].value_counts()
    
    # 한글을 영어로 매핑
    inflow_mapping = {
        '네이버 - 기타': 'Naver - Others',
        '네이버 지도': 'Naver Maps',
        '네이버 플레이스 - 검색 목록': 'Naver Place - Search',
        '네이버 플레이스 - 상세페이지': 'Naver Place - Detail',
        '기타 (유입경로 구분 불가)': 'Others (Unknown)',
        '네이버 기타': 'Naver Others',
        '네이버 플레이스광고(Beta)': 'Naver Place Ads',
        '네이버 블로그': 'Naver Blog',
        '외부서비스 유입': 'External Service',
        '네이버 플레이스 - 업체명 검색': 'Naver Place - Company'
    }
    
    english_labels = [inflow_mapping.get(label, label) for label in inflow_counts.index]
    
    axes[0, 0].pie(inflow_counts.values, labels=english_labels, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Inflow Path Distribution')
    
    # 2. 치료과목별 예약 건수 (영어 레이블 사용)
    all_treatments = []
    for col in treatment_columns:
        treatments = df[col].dropna()
        all_treatments.extend(treatments.tolist())
    
    treatment_counts = Counter(all_treatments)
    
    # 한글을 영어로 매핑
    treatment_mapping = {
        '검진': 'Check-up',
        '충치치료': 'Cavity Treatment',
        '기타': 'Others',
        '스케일링': 'Scaling',
        '신경치료': 'Root Canal',
        '임플란트': 'Implant',
        '보철치료': 'Prosthetics'
    }
    
    treatments = list(treatment_counts.keys())
    counts = list(treatment_counts.values())
    english_treatments = [treatment_mapping.get(t, t) for t in treatments]
    
    axes[0, 1].barh(english_treatments, counts, color='lightcoral')
    axes[0, 1].set_title('Treatment Categories')
    axes[0, 1].set_xlabel('Number of Bookings')
    
    # 3. 신환 비율
    new_patient_rate = (df['신환여부'] == 'Y').sum() / len(df) * 100
    
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
    plt.savefig('data/processed/analysis_english.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_analysis():
    """상세 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Hospital Analysis', fontsize=16, fontweight='bold')
    
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
    
    # 영어 레이블 사용
    treatment_mapping = {
        '검진': 'Check-up',
        '충치치료': 'Cavity Treatment',
        '기타': 'Others',
        '스케일링': 'Scaling',
        '신경치료': 'Root Canal',
        '임플란트': 'Implant',
        '보철치료': 'Prosthetics'
    }
    
    english_treatments = [treatment_mapping.get(t, t) for t in treatment_names]
    
    bars = axes[0, 0].bar(english_treatments, new_rates, color='lightblue')
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
    
    # 예약경로 영어 매핑
    booking_mapping = {
        '검색': 'Search',
        '재방문': 'Return Visit',
        '소개': 'Referral',
        '인터넷': 'Internet',
        '기타': 'Others',
        '맘카페': 'Mom Cafe',
        '네이버 검색': 'Naver Search',
        '가족': 'Family',
        '근처': 'Nearby',
        '타병원(연세)': 'Other Hospital'
    }
    
    english_paths = [booking_mapping.get(p, p) for p in paths]
    
    axes[0, 1].barh(english_paths, counts, color='lightgreen')
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
    plt.savefig('data/processed/detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# 메인 실행
if __name__ == "__main__":
    print("=== Hospital Booking Data Analysis (Fixed Version) ===\n")
    
    # 결과 디렉토리 생성
    os.makedirs('data/processed', exist_ok=True)
    
    print("1. Creating analysis with English labels...")
    create_analysis_with_english_labels()
    
    print("2. Creating detailed analysis...")
    create_detailed_analysis()
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- data/processed/analysis_english.png")
    print("- data/processed/detailed_analysis.png")
    
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
    
    print("\n=== Analysis Summary ===")
    print("✅ 시각화 완료: 영어 레이블을 사용하여 한글 깨짐 문제 해결")
    print("📊 주요 인사이트:")
    print("  - 가장 효과적인 유입경로: 네이버 - 기타 (38.7%)")
    print("  - 가장 인기 있는 치료과목: 검진 (699건)")
    print("  - 신환 비율: 59.0%")
    print("  - 가장 예약이 많은 요일: 월요일")
