import json
import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import shutil

def parse_date(date_str):
    """날짜 문자열을 파싱하여 datetime 객체로 변환"""
    try:
        # "2025.06.03" 형태의 날짜를 파싱
        return datetime.strptime(date_str, "%Y.%m.%d")
    except:
        # 파싱 실패시 최대값 반환 (우선순위 낮춤)
        return datetime.max

def remove_duplicates():
    """중복 게시글을 제거하고 가장 앞선 날짜의 게시글만 남깁니다."""
    
    # historical_processed 폴더 경로
    data_dir = Path("blog_automation/cafe_analysis/data/historical_processed")
    
    # 백업 폴더 생성
    backup_dir = data_dir / "backup_before_dedup"
    backup_dir.mkdir(exist_ok=True)
    
    # 모든 JSON 파일 찾기
    json_files = list(data_dir.glob("*.json"))
    print(f"총 {len(json_files)}개의 JSON 파일을 분석합니다...")
    
    # club_id와 article_id 조합별로 모든 게시글 수집
    all_articles = defaultdict(list)  # (club_id, article_id) -> [게시글 정보] 리스트
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
                            all_articles[key].append({
                                'file': file_path.name,
                                'index': idx,
                                'article': article,
                                'date': article.get('date', ''),
                                'parsed_date': parse_date(article.get('date', ''))
                            })
                            total_articles += 1
                        
        except Exception as e:
            print(f"파일 {file_path.name} 처리 중 오류: {e}")
    
    print(f"\n총 {total_articles}개의 아티클을 분석했습니다.")
    
    # 중복 건 찾기
    duplicates = {key: entries for key, entries in all_articles.items() if len(entries) > 1}
    
    print(f"\n중복 건 수: {len(duplicates)}")
    
    # 각 파일별로 제거할 인덱스 추적
    files_to_remove = defaultdict(set)
    
    # 중복 제거 로직
    for (club_id, article_id), entries in duplicates.items():
        # 날짜순으로 정렬 (가장 앞선 날짜가 먼저)
        sorted_entries = sorted(entries, key=lambda x: x['parsed_date'])
        
        # 첫 번째(가장 앞선 날짜)를 제외하고 나머지는 제거 대상
        for entry in sorted_entries[1:]:
            files_to_remove[entry['file']].add(entry['index'])
    
    print(f"\n제거할 중복 게시글 수: {sum(len(indices) for indices in files_to_remove.values())}")
    
    # 파일별로 중복 제거
    processed_files = 0
    total_removed = 0
    
    for file_path in json_files:
        if file_path.name in files_to_remove:
            try:
                # 백업 생성
                backup_path = backup_dir / file_path.name
                shutil.copy2(file_path, backup_path)
                
                # 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # 제거할 인덱스를 역순으로 정렬 (뒤에서부터 제거)
                    indices_to_remove = sorted(files_to_remove[file_path.name], reverse=True)
                    
                    # 중복 제거
                    for idx in indices_to_remove:
                        if idx < len(data):
                            del data[idx]
                            total_removed += 1
                    
                    # 파일 다시 쓰기
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    
                    processed_files += 1
                    print(f"파일 {file_path.name}에서 {len(indices_to_remove)}개 중복 제거")
                
            except Exception as e:
                print(f"파일 {file_path.name} 처리 중 오류: {e}")
    
    print(f"\n=== 중복 제거 완료 ===")
    print(f"처리된 파일 수: {processed_files}")
    print(f"제거된 중복 게시글 수: {total_removed}")
    print(f"백업 폴더: {backup_dir}")
    
    # 중복 제거 후 재검증
    print(f"\n=== 중복 제거 후 재검증 ===")
    verify_duplicates()

def verify_duplicates():
    """중복 제거 후 재검증"""
    data_dir = Path("blog_automation/cafe_analysis/data/historical_processed")
    json_files = list(data_dir.glob("*.json"))
    
    all_combinations = defaultdict(list)
    total_articles = 0
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for idx, article in enumerate(data):
                    if isinstance(article, dict):
                        club_id = article.get('club_id')
                        article_id = article.get('article_id')
                        
                        if club_id is not None and article_id is not None:
                            key = (club_id, article_id)
                            all_combinations[key].append({
                                'file': file_path.name,
                                'index': idx,
                                'article': article
                            })
                            total_articles += 1
                        
        except Exception as e:
            print(f"파일 {file_path.name} 처리 중 오류: {e}")
    
    duplicates = {key: entries for key, entries in all_combinations.items() if len(entries) > 1}
    
    print(f"총 아티클 수: {total_articles}")
    print(f"남은 중복 건 수: {len(duplicates)}")
    
    if duplicates:
        print("\n⚠️ 아직 중복이 남아있습니다:")
        for (club_id, article_id), entries in duplicates.items():
            print(f"  - club_id: {club_id}, article_id: {article_id} ({len(entries)}개)")
    else:
        print("✅ 모든 중복이 제거되었습니다!")

if __name__ == "__main__":
    remove_duplicates() 