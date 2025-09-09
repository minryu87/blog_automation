import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('data/raw/booking_record_detail.csv')

print("=== 병원 예약 데이터 통계 인사이트 ===\n")

# 날짜 파싱 함수 (수정된 버전)
def parse_datetime(date_str):
    if pd.isna(date_str):
        return pd.NaT
    
    # 패턴 매칭
    patterns = [
        r'(\d{4}-\d{2}-\d{2}) \(([월화수목금토일])\) 오전 (\d{1,2}):(\d{2}):(\d{2})',
        r'(\d{4}-\d{2}-\d{2}) \(([월화수목금토일])\) 오후 (\d{1,2}):(\d{2}):(\d{2})'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, str(date_str))
        if match:
            date_part, weekday, hour, minute, second = match.groups()
            hour = int(hour)
            
            # 오후인 경우 12시간 추가 (단, 12시는 그대로)
            if '오후' in str(date_str) and hour != 12:
                hour += 12
            elif '오전' in str(date_str) and hour == 12:
                hour = 0
            
            # 24시간 형식으로 변환
            hour = hour % 24
            
            return pd.to_datetime(f"{date_part} {hour:02d}:{minute}:{second}")
    
    return pd.NaT

# 날짜 파싱 적용
df['예약신청일시'] = df['예약신청일시'].apply(parse_datetime)

print("=== 1. 데이터 기본 정보 ===")
print(f"📊 총 예약 건수: {len(df):,}건")
print(f"📅 데이터 기간: {df['예약신청일시'].min().strftime('%Y-%m-%d')} ~ {df['예약신청일시'].max().strftime('%Y-%m-%d')}")
print(f"🆕 신환 비율: {(df['신환여부'] == 'Y').sum() / len(df) * 100:.1f}%")
print(f"✅ 이용완료 비율: {(df['상태'] == '이용완료').sum() / len(df) * 100:.1f}%")
print(f"❌ 취소 비율: {(df['상태'] == '취소').sum() / len(df) * 100:.1f}%")

print("\n=== 2. 유입경로 분석 ===")
inflow_counts = df['유입경로'].value_counts()
print("🔍 유입경로별 예약 건수 (상위 5개):")
for i, (path, count) in enumerate(inflow_counts.head().items(), 1):
    print(f"  {i}. {path}: {count:,}건 ({count/len(df)*100:.1f}%)")

print("\n=== 3. 치료과목 분석 ===")
treatment_columns = ['치료과목1', '치료과목2', '치료과목3', '치료과목4', '치료과목5', '치료과목6']
all_treatments = []
for col in treatment_columns:
    treatments = df[col].dropna()
    all_treatments.extend(treatments.tolist())

treatment_counts = Counter(all_treatments)
print("🦷 치료과목별 예약 건수:")
for treatment, count in treatment_counts.most_common():
    if treatment:
        print(f"  {treatment}: {count:,}건")

print("\n=== 4. 예약경로 분석 ===")
booking_columns = ['예약 경로1', '예약 경로2', '예약 경로3', '예약 경로4', '예약 경로5']
all_booking_paths = []
for col in booking_columns:
    paths = df[col].dropna()
    all_booking_paths.extend(paths.tolist())

booking_counts = Counter(all_booking_paths)
print("📞 예약경로별 건수 (상위 10개):")
for i, (path, count) in enumerate(booking_counts.most_common(10), 1):
    if path:
        print(f"  {i}. {path}: {count:,}건")

print("\n=== 5. 치료과목별 유입경로 분석 ===")
treatment_inflow = {}
for idx, row in df.iterrows():
    treatments = [row[col] for col in treatment_columns if pd.notna(row[col]) and row[col]]
    inflow = row['유입경로']
    
    for treatment in treatments:
        if treatment not in treatment_inflow:
            treatment_inflow[treatment] = Counter()
        treatment_inflow[treatment][inflow] += 1

print("💡 주요 치료과목별 유입경로 (상위 3개):")
for treatment, inflows in treatment_inflow.items():
    print(f"\n{treatment}:")
    for inflow, count in inflows.most_common(3):
        total = sum(inflows.values())
        print(f"  {inflow}: {count:,}건 ({count/total*100:.1f}%)")

print("\n=== 6. 치료과목별 신환비율 분석 ===")
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

print("🆕 치료과목별 신환 비율:")
for treatment, stats in treatment_new_patient.items():
    new_rate = stats['new'] / stats['total'] * 100
    print(f"  {treatment}: {stats['new']:,}/{stats['total']:,}건 ({new_rate:.1f}%)")

print("\n=== 7. 유입경로별 예약경로 분석 ===")
inflow_booking = {}
for idx, row in df.iterrows():
    inflow = row['유입경로']
    booking_paths = [row[col] for col in booking_columns if pd.notna(row[col]) and row[col]]
    
    if inflow not in inflow_booking:
        inflow_booking[inflow] = Counter()
    for path in booking_paths:
        inflow_booking[inflow][path] += 1

print("🔄 주요 유입경로별 예약경로 (상위 3개):")
for inflow, paths in inflow_booking.items():
    if sum(paths.values()) > 50:  # 50건 이상인 유입경로만
        print(f"\n{inflow}:")
        for path, count in paths.most_common(3):
            total = sum(paths.values())
            print(f"  {path}: {count:,}건 ({count/total*100:.1f}%)")

print("\n=== 8. 신환여부별 상세 분석 ===")
for new_patient in ['Y', 'N']:
    subset = df[df['신환여부'] == new_patient]
    patient_type = "신환" if new_patient == 'Y' else "재방문"
    
    print(f"\n{patient_type} ({len(subset):,}건):")
    
    # 유입경로
    inflow_counts = subset['유입경로'].value_counts()
    print("  🔍 주요 유입경로:")
    for path, count in inflow_counts.head(3).items():
        print(f"    {path}: {count:,}건 ({count/len(subset)*100:.1f}%)")
    
    # 치료과목
    all_treatments = []
    for col in treatment_columns:
        treatments = subset[col].dropna()
        all_treatments.extend(treatments.tolist())
    
    treatment_counts = Counter(all_treatments)
    print("  🦷 주요 치료과목:")
    for treatment, count in treatment_counts.most_common(3):
        if treatment:
            print(f"    {treatment}: {count:,}건")

print("\n=== 9. 시계열 분석 ===")
# 날짜별 예약 건수
daily_bookings = df.groupby(df['예약신청일시'].dt.date).size()
print(f"📈 일평균 예약 건수: {daily_bookings.mean():.1f}건")
print(f"📊 최대 일일 예약 건수: {daily_bookings.max()}건")
print(f"📉 최소 일일 예약 건수: {daily_bookings.min()}건")

# 요일별 분석
df['요일'] = df['예약신청일시'].dt.day_name()
weekday_counts = df['요일'].value_counts()
print("\n📅 요일별 예약 건수:")
for day, count in weekday_counts.items():
    print(f"  {day}: {count:,}건")

# 월별 분석
df['월'] = df['예약신청일시'].dt.month
monthly_bookings = df.groupby('월').size()
print("\n📆 월별 예약 건수:")
for month, count in monthly_bookings.items():
    print(f"  {month}월: {count:,}건")

print("\n=== 10. 주요 인사이트 요약 ===")
print("🎯 핵심 발견사항:")

# 1. 가장 효과적인 유입경로
top_inflow = inflow_counts.index[0]
print(f"  1. 가장 효과적인 유입경로: {top_inflow} ({inflow_counts.iloc[0]:,}건)")

# 2. 가장 인기 있는 치료과목
top_treatment = treatment_counts.most_common(1)[0][0]
print(f"  2. 가장 인기 있는 치료과목: {top_treatment} ({treatment_counts[top_treatment]:,}건)")

# 3. 신환 비율이 가장 높은 치료과목
max_new_rate_treatment = max(treatment_new_patient.items(), key=lambda x: x[1]['new']/x[1]['total'])
print(f"  3. 신환 비율이 가장 높은 치료과목: {max_new_rate_treatment[0]} ({max_new_rate_treatment[1]['new']/max_new_rate_treatment[1]['total']*100:.1f}%)")

# 4. 가장 많이 사용되는 예약경로
top_booking_path = booking_counts.most_common(1)[0][0]
print(f"  4. 가장 많이 사용되는 예약경로: {top_booking_path} ({booking_counts[top_booking_path]:,}건)")

# 5. 요일별 패턴
top_weekday = weekday_counts.index[0]
print(f"  5. 가장 예약이 많은 요일: {top_weekday} ({weekday_counts.iloc[0]:,}건)")

print("\n=== 11. 마케팅 전략 제안 ===")
print("📈 데이터 기반 마케팅 인사이트:")

# 유입경로별 전략
print("\n🔍 유입경로별 마케팅 전략:")
print("  • 네이버 - 기타 (38.7%): 가장 효과적인 유입경로로, 네이버 내 다양한 채널 활용 강화")
print("  • 네이버 지도 (20.2%): 지역 기반 검색 최적화 및 지도 리뷰 관리 중요")
print("  • 네이버 플레이스 (26.1%): 플레이스 페이지 최적화 및 광고 활용")

# 치료과목별 전략
print("\n🦷 치료과목별 마케팅 전략:")
print("  • 검진 (699건): 가장 인기 있는 서비스로, 정기검진 캠페인 강화")
print("  • 충치치료 (457건): 긴급 치료 서비스로, 즉시 예약 가능성 강조")
print("  • 스케일링 (358건): 예방적 치료로, 정기 관리의 중요성 홍보")

# 신환 확보 전략
print("\n🆕 신환 확보 전략:")
print("  • 스케일링 신환비율 69.8%: 예방적 치료로 신환 유입 효과적")
print("  • 검진 신환비율 57.9%: 정기검진으로 신환 확보 가능")
print("  • 신경치료 신환비율 54.5%: 전문 치료로 신환 유입")

print("\n=== 분석 완료 ===")
