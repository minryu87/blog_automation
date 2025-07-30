import json
import os
from collections import defaultdict
from pathlib import Path

def check_duplicates():
    """모든 JSON 파일에서 club_id와 article_id가 모두 동일한 중복 건을 찾습니다."""
    
    # historical_processed 폴더 경로
    data_dir = Path("blog_automation/cafe_analysis/data/historical_processed")
    
    # 모든 JSON 파일 찾기
    json_files = list(data_dir.glob("*.json"))
    print(f"총 {len(json_files)}개의 JSON 파일을 분석합니다...")
    
    # club_id와 article_id 조합을 추적
    all_combinations = defaultdict(list)  # (club_id, article_id) -> [파일명, 인덱스] 리스트
    total_articles = 0
    
    # 각 파일을 순회하면서 분석
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # data가 배열인지 확인
            if isinstance(data, list):
                for idx, article in enumerate(data):
                    if isinstance(article, dict):
                        club_id = article.get('club_id')
                        article_id = article.get('article_id')
                        
                        if club_id is not None and article_id is not None:
                            # club_id와 article_id 조합을 키로 사용
                            key = (club_id, article_id)
                            all_combinations[key].append({
                                'file': file_path.name,
                                'index': idx,
                                'article': article
                            })
                            total_articles += 1
                        
        except Exception as e:
            print(f"파일 {file_path.name} 처리 중 오류: {e}")
    
    print(f"\n총 {total_articles}개의 아티클을 분석했습니다.")
    
    # 중복 건 찾기
    duplicates = {key: entries for key, entries in all_combinations.items() if len(entries) > 1}
    
    print(f"\n중복 건 수: {len(duplicates)}")
    
    if duplicates:
        print("\n=== 중복 상세 내역 ===")
        for (club_id, article_id), entries in duplicates.items():
            print(f"\nclub_id: {club_id}, article_id: {article_id}")
            print(f"중복 횟수: {len(entries)}")
            for entry in entries:
                print(f"  - 파일: {entry['file']}, 인덱스: {entry['index']}")
                # 제목이 있다면 출력
                title = entry['article'].get('title', '제목 없음')
                print(f"    제목: {title[:100]}...")
    
    # 통계 요약
    if total_articles > 0:
        total_duplicate_articles = sum(len(entries) for entries in duplicates.values())
        unique_articles = total_articles - total_duplicate_articles + len(duplicates)
        
        print(f"\n=== 통계 요약 ===")
        print(f"총 아티클 수: {total_articles}")
        print(f"고유 아티클 수: {unique_articles}")
        print(f"중복 건 수: {len(duplicates)}")
        print(f"중복된 아티클 수: {total_duplicate_articles}")
        print(f"중복률: {(total_duplicate_articles / total_articles * 100):.2f}%")
    else:
        print("\n분석할 아티클이 없습니다.")

if __name__ == "__main__":
    check_duplicates() 