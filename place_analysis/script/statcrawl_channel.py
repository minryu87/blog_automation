#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
네이버 스마트플레이스 채널별 PV 데이터 수집 스크립트
2025년 7월 한 달간의 데이터를 수집하여 테이블로 저장
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import os
from typing import Dict, List, Any

class NaverSmartPlaceCollector:
    def __init__(self):
        """초기화"""
        self.base_url = "https://new.smartplace.naver.com/proxy/bizadvisor/api/v3/sites/sp_311b9ba993e974/report"
        
        # Authorization 헤더
        self.auth_token = """Bearer slEQJHjSWy8/nh/Eeb+JT+sPFvVIbXGt1vgnCzgibRCQ1nbI99oQvX+4866WDnWohN6XEnKnxjcx5Gqlw08W80XrCTBzIkyQaYn39Yw/4DFGm00iJI5aj9TXvlacwUE+h8Q28V6gCtEjFULG5gieGkoNLtk6h87/O1pvOSstE2PQ6R5Hze8kPYn4FmIUJSP+73Me4z0RYWuzw1kCDaaQKnkqBrVdSMwj2X33q8J3MJe+Tnoxda2OezO0dy+fl/vDznV68RulT/sGcJhzxCOmVqEnjAHkp3x00s/fCubLG5oRPtXNtEjziiGwzQo9tPgWrX0LLk0nEbuNU0isIh6Y2RDBjsIRGEnE87LAoqKJ3Lw="""
        
        # Cookie
        self.cookie = """NNB=ZKHGWSJXM6SGO; NSCS=1; ASID=01dcd28c00000194f85124240000004f; NAC=LjaaBowe2zA0A; JSESSIONID=5260E43AD131D5D717A845CBEB327C63; nid_inf=1760525603; NID_AUT=GUytzV9XzHD04v3E4zcypSLif601MbzHbOIWXMj3Zr6vk6eVVOYrcNGnNWDjKdLf; page_uid=j5divlqo15VssTyVvJCssssstEZ-057330; csrf_token=39773df8ad4251958c25719ac5c7f46c06d86b2e5d22c5709ab225d8d9bf73ffde594d7935800c2cdac78bd95e33e66ee0facc523dca0cd7f0b414b1b4a07836; NACT=1; SRT30=1754554830; BUC=1H_paitS7sce4fyPW6Npbe9whz05C7MDpTHGgoVT1nQ=; NID_SES=AAABjEsr8/1iTsnCC3nrV3NJLx+w71fwztrvXYTekr0kjxb+Q42Wg2zR9BQS/UibrD35TRVc3dEmICRVD2lTnFv7bz/EGHFDmUqnPw/zTdNs57Asicu25U4ONiUGEWPA2eJHsdYvECiLDdchVDEt7Myzcr4Wjv51olROiCuxqiBrhhDwUU6rmkAw3maLUnX333S5ZyYcitpxGb0OSCcYugpihV0TJZVev41ttPEt10NCDfvxPWqkPfTflQZ/hLKmbcrVQExv457vsR9nn5io3X5SlLEj/ZBHIi/fV43SoZjoKUg/RVOoKyVCC1Qt+4EUSommRVtw84prIIHj0VseyB3O4lVVaZHUlxUe/wdQ5qcOVUtf2WAhyCqVqnfs3r9fo2RwC+yt3pLxvWbN4bCrFOdPu4ImQEtaEk7KQJl+3DXoNy5vUgAHBSgGbCGOQ51H2vWMJmaOX5TmwwWPUjEQwy2Kf73GSPMi3hjgIyNHK7960XqdpMIpsh+2rv2GMhShlVBuc3G/VH2/lNEfKmILtP/K9Tk=; ba_access_token=slEQJHjSWy8%2Fnh%2FEeb%2BJT%2BsPFvVIbXGt1vgnCzgibRCQ1nbI99oQvX%2B4866WDnWohN6XEnKnxjcx5Gqlw08W80XrCTBzIkyQaYn39Yw%2F4DFGm00iJI5aj9TXvlacwUE%2Bh8Q28V6gCtEjFULG5gieGkoNLtk6h87%2FO1pvOSstE2PQ6R5Hze8kPYn4FmIUJSP%2B73Me4z0RYWuzw1kCDaaQKnkqBrVdSMwj2X33q8J3MJe%2BTnoxda2OezO0dy%2Bfl%2FvDznV68RulT%2FsGcJhzxCOmVqEnjAHkp3x00s%2FfCubLG5oRPtXNtEjziiGwzQo9tPgWrX0LLk0nEbuNU0isIh6Y2RDBjsIRGEnE87LAoqKJ3Lw%3D"""
        
        self.headers = {
            'Authorization': self.auth_token,
            'Cookie': self.cookie,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://new.smartplace.naver.com/'
        }
        
        self.all_data = {}
        self.channels = set()

    def fetch_data_for_date(self, date: str) -> List[Dict]:
        """특정 날짜의 채널별 데이터를 가져오기"""
        params = {
            'dimensions': 'mapped_channel_name',  # 채널별로 그룹화
            'startDate': date,
            'endDate': date,
            'metrics': 'pv',
            'sort': 'pv',
            'useIndex': 'revenue-all-channel-detail'  # 전체 채널 상세
        }
        
        try:
            print(f"📊 {date} 데이터 수집 중...", end=' ')
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # 수집된 채널 목록 표시
                channels_found = [item['mapped_channel_name'] for item in data if 'mapped_channel_name' in item]
                print(f"✅ {len(data)}개 채널 ({', '.join(channels_found[:3])}{'...' if len(channels_found) > 3 else ''})")
                
                # 첫 날짜의 데이터 구조 확인
                if date == '2025-07-01' and data:
                    print(f"\n📌 데이터 구조 확인:")
                    for i, item in enumerate(data, 1):
                        print(f"   {i}. {json.dumps(item, ensure_ascii=False)}")
                    print()
                
                return data
            else:
                print(f"⚠️ HTTP {response.status_code} 에러")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 네트워크 에러 - {str(e)}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 에러 - {str(e)}")
            return []

    def collect_all_data(self):
        """2025년 7월 전체 데이터 수집"""
        print("=" * 60)
        print("🚀 네이버 스마트플레이스 채널별 PV 데이터 수집 시작")
        print("📅 기간: 2025년 7월 1일 ~ 31일")
        print("=" * 60)
        
        start_date = datetime(2025, 7, 1)
        end_date = datetime(2025, 7, 31)
        
        # 먼저 모든 데이터 수집
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # API 호출
            data = self.fetch_data_for_date(date_str)
            
            # 데이터 저장
            self.all_data[date_str] = data
            
            # 다음 날짜로
            current_date += timedelta(days=1)
            
            # API 부하 방지를 위한 딜레이
            time.sleep(0.5)
        
        # 모든 데이터 수집 후 전체 채널 목록 파악
        print("\n📊 전체 채널 목록 파악 중...")
        channel_appearances = {}  # 채널별 출현 횟수
        
        for date_str, data_list in self.all_data.items():
            for item in data_list:
                if 'mapped_channel_name' in item and item['mapped_channel_name']:
                    channel_name = item['mapped_channel_name']
                    self.channels.add(channel_name)
                    
                    # 채널별 출현 횟수 카운트
                    if channel_name not in channel_appearances:
                        channel_appearances[channel_name] = 0
                    channel_appearances[channel_name] += 1
        
        print(f"\n✅ 데이터 수집 완료!")
        print(f"📊 총 {len(self.channels)}개 채널 발견:")
        
        # 출현 빈도순으로 정렬하여 출력
        sorted_channels = sorted(channel_appearances.items(), key=lambda x: x[1], reverse=True)
        for channel, count in sorted_channels:
            print(f"   - {channel}: {count}일 출현")

    def create_dataframe(self) -> pd.DataFrame:
        """수집된 데이터를 DataFrame으로 변환"""
        if not self.all_data:
            print("⚠️ 수집된 데이터가 없습니다.")
            return pd.DataFrame()
        
        # 채널 목록이 비어있으면 다시 스캔
        if not self.channels:
            print("\n🔄 채널 정보 재스캔 중...")
            for date_str, data_list in self.all_data.items():
                for item in data_list:
                    if 'mapped_channel_name' in item and item['mapped_channel_name']:
                        self.channels.add(item['mapped_channel_name'])
            print(f"   재스캔 완료: {len(self.channels)}개 채널 발견")
        
        if not self.channels:
            print("⚠️ 채널 정보를 찾을 수 없습니다.")
            # 디버깅을 위해 첫 번째 날짜의 데이터 구조 출력
            first_date = sorted(self.all_data.keys())[0] if self.all_data else None
            if first_date and self.all_data[first_date]:
                print(f"\n🔍 {first_date} 데이터 샘플:")
                for item in self.all_data[first_date][:3]:
                    print(f"   {json.dumps(item, ensure_ascii=False)}")
            return pd.DataFrame()
        
        # 날짜별로 정렬
        sorted_dates = sorted(self.all_data.keys())
        
        # DataFrame 생성
        print(f"\n📊 데이터프레임 생성 중... (날짜: {len(sorted_dates)}일, 채널: {len(self.channels)}개)")
        data_for_df = []
        
        for date in sorted_dates:
            row = {'날짜': date}
            
            # 각 채널별 PV 값 설정
            for channel in self.channels:
                # 해당 날짜의 채널 데이터 찾기
                channel_data = None
                for item in self.all_data[date]:
                    if item.get('mapped_channel_name') == channel:
                        channel_data = item
                        break
                
                # PV 값 추출 (float 형태로 저장된 경우 처리)
                if channel_data:
                    pv_value = channel_data.get('pv', 0)
                    # float인 경우 정수로 변환
                    row[channel] = int(pv_value) if isinstance(pv_value, float) else pv_value
                else:
                    row[channel] = 0
            
            data_for_df.append(row)
        
        df = pd.DataFrame(data_for_df)
        
        # 날짜를 인덱스로 설정
        df.set_index('날짜', inplace=True)
        
        # 채널을 PV 총합 기준으로 정렬
        channel_totals = df.sum().sort_values(ascending=False)
        df = df[channel_totals.index]
        
        print(f"   생성 완료: {df.shape[0]}행 × {df.shape[1]}열")
        
        return df

    def calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """통계 요약 계산"""
        if df.empty:
            return pd.DataFrame()
        
        stats = pd.DataFrame({
            '총합': df.sum(),
            '일평균': df.mean().round(1),
            '최대': df.max(),
            '최소': df.min(),
            '표준편차': df.std().round(1)
        })
        
        # PV 총합 기준으로 정렬
        stats = stats.sort_values('총합', ascending=False)
        
        return stats.T

    def save_to_files(self, df: pd.DataFrame, stats: pd.DataFrame):
        """결과를 파일로 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV 파일 저장
        csv_filename = f'channel_pv_data_2025_07_{timestamp}.csv'
        df.to_csv(csv_filename, encoding='utf-8-sig')
        print(f"\n📁 CSV 파일 저장: {csv_filename}")
        
        # Excel 파일 저장
        try:
            excel_filename = f'channel_pv_data_2025_07_{timestamp}.xlsx'
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='일별 데이터')
                stats.to_excel(writer, sheet_name='통계 요약')
                
                # 월간 트렌드 차트용 데이터 추가
                monthly_trend = df.T
                monthly_trend.to_excel(writer, sheet_name='채널별 트렌드')
                
            print(f"📁 Excel 파일 저장: {excel_filename}")
        except ImportError:
            print("⚠️ Excel 저장을 위해 openpyxl 설치가 필요합니다: pip install openpyxl")
        except Exception as e:
            print(f"⚠️ Excel 저장 실패: {str(e)}")
        
        # JSON 파일 저장
        json_filename = f'channel_pv_raw_data_2025_07_{timestamp}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)
        print(f"📁 JSON 원본 데이터 저장: {json_filename}")

    def display_results(self, df: pd.DataFrame, stats: pd.DataFrame):
        """결과 출력"""
        print("\n" + "=" * 80)
        print("📊 2025년 7월 채널별 PV 데이터")
        print("=" * 80)
        
        # DataFrame 출력 설정
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 20)
        pd.set_option('display.float_format', lambda x: '%.0f' % x)
        
        # 처음과 마지막 일부만 출력
        if len(df) > 15:
            print("\n[처음 7일]")
            print(df.head(7))
            print("\n   ... 중간 생략 ...\n")
            print("[마지막 7일]")
            print(df.tail(7))
        else:
            print(df)
        
        print("\n" + "=" * 80)
        print("📈 채널별 통계 요약")
        print("=" * 80)
        print(stats)
        
        # 주요 인사이트
        print("\n" + "=" * 80)
        print("🎯 주요 인사이트")
        print("=" * 80)
        
        # 총 PV 합계
        total_pv = df.sum().sum()
        print(f"📍 전체 PV 총합: {total_pv:,.0f}")
        
        # 일평균 PV
        daily_avg = total_pv / len(df)
        print(f"📍 일평균 전체 PV: {daily_avg:,.1f}")
        
        # 가장 높은 PV 채널
        if not df.empty and len(df.columns) > 0:
            best_channel = df.sum().idxmax()
            best_channel_pv = df.sum().max()
            best_channel_pct = (best_channel_pv / total_pv * 100)
            print(f"📍 최고 성과 채널: {best_channel} ({best_channel_pv:,.0f} PV, {best_channel_pct:.1f}%)")
            
            # 상위 3개 채널
            top3 = df.sum().nlargest(3)
            print(f"\n📍 TOP 3 채널:")
            for i, (channel, pv) in enumerate(top3.items(), 1):
                pct = (pv / total_pv * 100)
                print(f"   {i}. {channel}: {pv:,.0f} PV ({pct:.1f}%)")
        
        # 가장 PV가 높았던 날
        daily_totals = df.sum(axis=1)
        best_day = daily_totals.idxmax()
        best_day_pv = daily_totals.max()
        print(f"\n📍 최고 PV 날짜: {best_day} ({best_day_pv:,.0f} PV)")
        
        # 가장 PV가 낮았던 날
        worst_day = daily_totals.idxmin()
        worst_day_pv = daily_totals.min()
        print(f"📍 최저 PV 날짜: {worst_day} ({worst_day_pv:,.0f} PV)")

    def create_visual_report(self, df: pd.DataFrame):
        """시각적 리포트 생성 (선택사항)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            
            # 한글 폰트 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('2025년 7월 채널별 PV 분석 리포트', fontsize=16)
            
            # 1. 일별 전체 PV 추이
            daily_totals = df.sum(axis=1)
            axes[0, 0].plot(range(len(daily_totals)), daily_totals.values, marker='o')
            axes[0, 0].set_title('일별 전체 PV 추이')
            axes[0, 0].set_xlabel('날짜')
            axes[0, 0].set_ylabel('PV')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 채널별 총 PV (파이 차트)
            channel_totals = df.sum().sort_values(ascending=False)
            axes[0, 1].pie(channel_totals.values, labels=channel_totals.index, autopct='%1.1f%%')
            axes[0, 1].set_title('채널별 PV 비중')
            
            # 3. 채널별 일별 추이 (상위 4개)
            top_channels = df.sum().nlargest(4).index
            for channel in top_channels:
                axes[1, 0].plot(range(len(df)), df[channel].values, marker='o', label=channel, alpha=0.7)
            axes[1, 0].set_title('주요 채널 일별 추이')
            axes[1, 0].set_xlabel('날짜')
            axes[1, 0].set_ylabel('PV')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 채널별 평균 PV (막대 그래프)
            channel_means = df.mean().sort_values(ascending=False)
            axes[1, 1].bar(range(len(channel_means)), channel_means.values)
            axes[1, 1].set_xticks(range(len(channel_means)))
            axes[1, 1].set_xticklabels(channel_means.index, rotation=45, ha='right')
            axes[1, 1].set_title('채널별 일평균 PV')
            axes[1, 1].set_ylabel('평균 PV')
            
            plt.tight_layout()
            
            # 파일로 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_filename = f'channel_pv_chart_2025_07_{timestamp}.png'
            plt.savefig(chart_filename, dpi=100, bbox_inches='tight')
            print(f"\n📁 차트 이미지 저장: {chart_filename}")
            
            plt.close()
            
        except ImportError:
            print("\n💡 차트 생성을 원하시면 matplotlib 설치가 필요합니다: pip install matplotlib")
        except Exception as e:
            print(f"\n⚠️ 차트 생성 중 오류: {str(e)}")

    def run(self):
        """전체 프로세스 실행"""
        # 1. 데이터 수집
        self.collect_all_data()
        
        # 2. DataFrame 생성
        df = self.create_dataframe()
        
        if df.empty:
            print("\n❌ 데이터 처리 실패")
            
            # 원본 데이터는 저장
            if self.all_data:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                json_filename = f'channel_pv_raw_data_2025_07_{timestamp}.json'
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(self.all_data, f, ensure_ascii=False, indent=2)
                print(f"📁 원본 JSON 데이터는 저장되었습니다: {json_filename}")
            return
        
        # 3. 통계 계산
        stats = self.calculate_statistics(df)
        
        # 4. 결과 출력
        self.display_results(df, stats)
        
        # 5. 파일 저장
        self.save_to_files(df, stats)
        
        # 6. 시각적 리포트 생성 (선택사항)
        self.create_visual_report(df)
        
        print("\n✨ 모든 작업이 완료되었습니다!")

def main():
    """메인 함수"""
    print("""
╔══════════════════════════════════════════════════════════╗
║     네이버 스마트플레이스 채널별 PV 데이터 수집기        ║
║                    Version 2.0                           ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 필요한 패키지 확인
    try:
        import requests
        import pandas as pd
    except ImportError as e:
        print("❌ 필요한 패키지가 설치되지 않았습니다.")
        print("다음 명령어로 설치해주세요:")
        print("pip install requests pandas openpyxl")
        print("\n차트 생성을 원하시면 추가로:")
        print("pip install matplotlib")
        return
    
    # 수집기 실행
    collector = NaverSmartPlaceCollector()
    
    # 사용자 확인
    print("현재 설정된 인증 정보로 데이터를 수집하시겠습니까?")
    print("(새로운 토큰이 있다면 코드의 auth_token과 cookie를 수정 후 실행하세요)")
    response = input("\n진행하시겠습니까? (y/n): ")
    
    if response.lower() == 'y':
        collector.run()
    else:
        print("🛑 작업이 취소되었습니다.")

if __name__ == "__main__":
    main()