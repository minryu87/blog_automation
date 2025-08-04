import os
import json
import shutil
import glob

def recover_empty_files():
    """
    빈 파일들을 백업에서 복구합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '..', 'data', 'historical_processed')
    backup_dir = os.path.join(processed_dir, 'backup_before_dedup')
    
    print("=== 빈 파일 복구 작업 시작 ===")
    
    if not os.path.exists(backup_dir):
        print("backup_before_dedup 폴더를 찾을 수 없습니다.")
        return
    
    # 빈 파일 목록 (check_empty_files.py 결과 기반)
    empty_files = [
        '2024-07_page_005_analyzed.json',
        '2024-12_page_005_analyzed.json',
        '2024-12_page_007_analyzed.json',
        '2024-12_page_008_analyzed.json',
        '2025-01_page_006_analyzed.json',
        '2025-01_page_008_analyzed.json',
        '2025-01_page_009_analyzed.json',
        '2025-01_page_010_analyzed.json',
        '2025-01_page_016_analyzed.json',
        '2025-02_page_004_analyzed.json',
        '2025-02_page_006_analyzed.json',
        '2025-02_page_007_analyzed.json',
        '2025-02_page_008_analyzed.json',
        '2025-03_page_005_analyzed.json',
        '2025-04_page_006_analyzed.json',
        '2025-04_page_009_analyzed.json',
        '2025-06_page_005_analyzed.json',
        '2025-06_page_006_analyzed.json',
        '2025-06_page_008_analyzed.json'
    ]
    
    recovered_count = 0
    failed_count = 0
    
    for filename in empty_files:
        backup_path = os.path.join(backup_dir, filename)
        target_path = os.path.join(processed_dir, filename)
        
        if os.path.exists(backup_path):
            try:
                # 백업 파일이 유효한지 확인
                with open(backup_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data and len(data) > 0:
                    # 백업 파일을 대상 위치로 복사
                    shutil.copy2(backup_path, target_path)
                    print(f"  복구 완료: {filename} ({len(data)}개 게시글)")
                    recovered_count += 1
                else:
                    print(f"  백업 파일도 비어있음: {filename}")
                    failed_count += 1
            except Exception as e:
                print(f"  복구 실패 ({filename}): {e}")
                failed_count += 1
        else:
            print(f"  백업 파일 없음: {filename}")
            failed_count += 1
    
    print(f"\n=== 복구 작업 완료 ===")
    print(f"복구 성공: {recovered_count}개")
    print(f"복구 실패: {failed_count}개")
    
    return recovered_count, failed_count

if __name__ == "__main__":
    recover_empty_files() 