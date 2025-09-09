import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('data/raw/booking_record_detail.csv')

print("=== 데이터 기본 정보 ===")
print(f"총 예약 건수: {len(df):,}건")
print(f"데이터 기간: {df['예약신청일시'].min()} ~ {df['예약신청일시'].max()}")
print("\n=== 컬럼 정보 ===")
print(df.columns.tolist())

# 기본 통계
print("\n=== 기본 통계 ===")
print(f"신환 비율: {(df['신환여부'] == 'Y').sum() / len(df) * 100:.1f}%")
print(f"이용완료 비율: {(df['상태'] == '이용완료').sum() / len(df) * 100:.1f}%")
print(f"취소 비율: {(df['상태'] == '취소').sum() / len(df) * 100:.1f}%")

# 유입경로 분석
print("\n=== 유입경로 분석 ===")
inflow_counts = df['유입경로'].value_counts()
print("유입경로별 예약 건수:")
for path, count in inflow_counts.items():
    print(f"  {path}: {count:,}건 ({count/len(df)*100:.1f}%)")

# 치료과목 분석
print("\n=== 치료과목 분석 ===")
treatment_columns = ['치료과목1', '치료과목2', '치료과목3', '치료과목4', '치료과목5', '치료과목6']
all_treatments = []
for col in treatment_columns:
    treatments = df[col].dropna()
    all_treatments.extend(treatments.tolist())

treatment_counts = Counter(all_treatments)
print("치료과목별 예약 건수:")
for treatment, count in treatment_counts.most_common():
    if treatment:  # 빈 값 제외
        print(f"  {treatment}: {count:,}건")

# 예약경로 분석
print("\n=== 예약경로 분석 ===")
booking_columns = ['예약 경로1', '예약 경로2', '예약 경로3', '예약 경로4', '예약 경로5']
all_booking_paths = []
for col in booking_columns:
    paths = df[col].dropna()
    all_booking_paths.extend(paths.tolist())

booking_counts = Counter(all_booking_paths)
print("예약경로별 건수:")
for path, count in booking_counts.most_common():
    if path:  # 빈 값 제외
        print(f"  {path}: {count:,}건")

# 치료과목별 유입경로 분석
print("\n=== 치료과목별 유입경로 분석 ===")
treatment_inflow = {}
for idx, row in df.iterrows():
    treatments = [row[col] for col in treatment_columns if pd.notna(row[col]) and row[col]]
    inflow = row['유입경로']
    
    for treatment in treatments:
        if treatment not in treatment_inflow:
            treatment_inflow[treatment] = Counter()
        treatment_inflow[treatment][inflow] += 1

for treatment, inflows in treatment_inflow.items():
    print(f"\n{treatment}:")
    total = sum(inflows.values())
    for inflow, count in inflows.most_common():
        print(f"  {inflow}: {count:,}건 ({count/total*100:.1f}%)")

# 치료과목별 예약경로 분석
print("\n=== 치료과목별 예약경로 분석 ===")
treatment_booking = {}
for idx, row in df.iterrows():
    treatments = [row[col] for col in treatment_columns if pd.notna(row[col]) and row[col]]
    booking_paths = [row[col] for col in booking_columns if pd.notna(row[col]) and row[col]]
    
    for treatment in treatments:
        if treatment not in treatment_booking:
            treatment_booking[treatment] = Counter()
        for path in booking_paths:
            treatment_booking[treatment][path] += 1

for treatment, paths in treatment_booking.items():
    print(f"\n{treatment}:")
    total = sum(paths.values())
    for path, count in paths.most_common():
        print(f"  {path}: {count:,}건 ({count/total*100:.1f}%)")

# 치료과목별 신환비율 분석
print("\n=== 치료과목별 신환비율 분석 ===")
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

for treatment, stats in treatment_new_patient.items():
    new_rate = stats['new'] / stats['total'] * 100
    print(f"{treatment}: {stats['new']:,}/{stats['total']:,}건 ({new_rate:.1f}%)")

# 유입경로별 예약경로 분석
print("\n=== 유입경로별 예약경로 분석 ===")
inflow_booking = {}
for idx, row in df.iterrows():
    inflow = row['유입경로']
    booking_paths = [row[col] for col in booking_columns if pd.notna(row[col]) and row[col]]
    
    if inflow not in inflow_booking:
        inflow_booking[inflow] = Counter()
    for path in booking_paths:
        inflow_booking[inflow][path] += 1

for inflow, paths in inflow_booking.items():
    print(f"\n{inflow}:")
    total = sum(paths.values())
    for path, count in paths.most_common():
        print(f"  {path}: {count:,}건 ({count/total*100:.1f}%)")

# 신환여부별 분석
print("\n=== 신환여부별 분석 ===")
for new_patient in ['Y', 'N']:
    subset = df[df['신환여부'] == new_patient]
    patient_type = "신환" if new_patient == 'Y' else "재방문"
    
    print(f"\n{patient_type} ({len(subset):,}건):")
    
    # 유입경로
    inflow_counts = subset['유입경로'].value_counts()
    print("  유입경로:")
    for path, count in inflow_counts.head(5).items():
        print(f"    {path}: {count:,}건 ({count/len(subset)*100:.1f}%)")
    
    # 예약경로
    all_booking_paths = []
    for col in booking_columns:
        paths = subset[col].dropna()
        all_booking_paths.extend(paths.tolist())
    
    booking_counts = Counter(all_booking_paths)
    print("  예약경로:")
    for path, count in booking_counts.most_common(5):
        if path:
            print(f"    {path}: {count:,}건")
    
    # 치료과목
    all_treatments = []
    for col in treatment_columns:
        treatments = subset[col].dropna()
        all_treatments.extend(treatments.tolist())
    
    treatment_counts = Counter(all_treatments)
    print("  치료과목:")
    for treatment, count in treatment_counts.most_common(5):
        if treatment:
            print(f"    {treatment}: {count:,}건")

# 시계열 분석
print("\n=== 시계열 분석 ===")
df['예약신청일시'] = pd.to_datetime(df['예약신청일시'], format='%Y-%m-%d (%a) 오후 %H:%M:%S', errors='coerce')
df['예약신청일시'] = pd.to_datetime(df['예약신청일시'], format='%Y-%m-%d (%a) 오전 %H:%M:%S', errors='coerce')

# 날짜별 예약 건수
daily_bookings = df.groupby(df['예약신청일시'].dt.date).size()
print(f"일평균 예약 건수: {daily_bookings.mean():.1f}건")
print(f"최대 일일 예약 건수: {daily_bookings.max()}건")
print(f"최소 일일 예약 건수: {daily_bookings.min()}건")

# 요일별 분석
df['요일'] = df['예약신청일시'].dt.day_name()
weekday_counts = df['요일'].value_counts()
print("\n요일별 예약 건수:")
for day, count in weekday_counts.items():
    print(f"  {day}: {count:,}건")

print("\n=== 분석 완료 ===")
