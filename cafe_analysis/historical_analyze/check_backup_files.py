import os
import json
import glob

def check_backup_files():
    """
    backup_before_dedup 폴더의 파일들을 확인합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backup_dir = os.path.join(script_dir, '..', 'data', 'historical_processed', 'backup_before_dedup')
    
    print("=== 백업 파일 확인 ===")
    
    if not os.path.exists(backup_dir):
        print("backup_before_dedup 폴더를 찾을 수 없습니다.")
        return
    
    # 모든 analyzed.json 파일 찾기
    json_files = glob.glob(os.path.join(backup_dir, '*_analyzed.json'))
    print(f"총 {len(json_files)}개의 백업 파일을 확인합니다.")
    
    backup_files = []
    total_files = 0
    
    for file_path in json_files:
        total_files += 1
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data and len(data) > 0:
                    backup_files.append({
                        'filename': filename,
                        'count': len(data)
                    })
                    print(f"  백업 파일: {filename} ({len(data)}개 게시글)")
                        
        except Exception as e:
            print(f"  오류 발생 ({filename}): {e}")
    
    print(f"\n=== 백업 파일 요약 ===")
    print(f"총 백업 파일 수: {total_files}")
    print(f"유효한 백업 파일 수: {len(backup_files)}")
    
    if backup_files:
        print(f"\n=== 백업 파일 목록 ===")
        for backup in sorted(backup_files, key=lambda x: x['filename']):
            print(f"  - {backup['filename']} ({backup['count']}개 게시글)")
    
    return backup_files

if __name__ == "__main__":
    check_backup_files() 