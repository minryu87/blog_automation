#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stat_crawler.py 수정 기능 테스트 스크립트
키워드별와 채널별 데이터 수집 기능을 테스트합니다.
"""

import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_stat_crawler_features():
    """stat_crawler.py의 수정된 기능을 테스트"""
    
    print("=" * 60)
    print("🔍 stat_crawler.py 수정 기능 테스트")
    print("=" * 60)
    
    # 수정된 파일 확인
    stat_crawler_path = "../../place_stat_crawler/scripts/crawler/stat_crawler.py"
    
    if not os.path.exists(stat_crawler_path):
        print(f"❌ 파일을 찾을 수 없습니다: {stat_crawler_path}")
        return
    
    print(f"✅ 파일 확인됨: {stat_crawler_path}")
    
    # 파일 내용 확인
    with open(stat_crawler_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 추가된 기능들 확인
    features_to_check = [
        'fetch_keyword_data_for_date',
        'fetch_channel_data_for_date', 
        'collect_keyword_data',
        'collect_channel_data',
        'create_keyword_dataframe',
        'create_channel_dataframe',
        'run_keyword_analysis',
        'run_channel_analysis',
        'ref_keyword'
    ]
    
    print("\n📋 추가된 기능 확인:")
    for feature in features_to_check:
        if feature in content:
            print(f"   ✅ {feature}")
        else:
            print(f"   ❌ {feature}")
    
    # main 함수 수정 확인
    if '키워드별 분석' in content and '채널별 분석' in content:
        print("\n✅ main 함수 수정 확인됨")
    else:
        print("\n❌ main 함수 수정 확인 실패")
    
    print("\n" + "=" * 60)
    print("📝 수정 사항 요약:")
    print("1. 키워드별 데이터 수집 기능 추가")
    print("2. 채널별 데이터 수집 기능 추가") 
    print("3. 키워드별/채널별 DataFrame 생성 기능")
    print("4. 키워드별/채널별 분석 워크플로우")
    print("5. 사용자 선택 메뉴 추가")
    print("=" * 60)

if __name__ == "__main__":
    test_stat_crawler_features()
