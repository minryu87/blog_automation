import os
import json
import logging
import time
import shutil
from tqdm import tqdm
from cafe_content_analyzer import CafePostAnalyzer, get_llm_agent

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories(base_dir):
    """파이프라인에 필요한 모든 디렉토리를 확인하고 없으면 생성합니다."""
    paths = {
        "raw": os.path.join(base_dir, 'data', 'historical_raw'),
        "processed": os.path.join(base_dir, 'data', 'historical_processed'),
        "archive": os.path.join(base_dir, 'data', 'historical_raw_archive')
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def process_single_file(file_path, analyzer, processed_dir, archive_dir):
    """단일 JSON 파일을 읽고, 분석하고, 결과 저장 및 원본 보관을 처리합니다."""
    logging.info(f"--- 처리 시작: {os.path.basename(file_path)} ---")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            posts = json.load(f)

        analyzed_posts = []
        # tqdm을 사용하여 개별 파일 내의 게시글 처리 진행 상황을 표시
        for post in tqdm(posts, desc=f"분석 중: {os.path.basename(file_path)}"):
            analysis_result = analyzer.analyze(post)
            post['analysis'] = analysis_result
            analyzed_posts.append(post)

        # 분석 완료된 데이터를 processed 폴더에 저장
        processed_filename = os.path.basename(file_path).replace('_processed_test.json', '_analyzed.json')
        processed_filepath = os.path.join(processed_dir, processed_filename)
        with open(processed_filepath, 'w', encoding='utf-8') as f:
            json.dump(analyzed_posts, f, ensure_ascii=False, indent=4)
        
        logging.info(f"분석 완료. 결과 저장: {processed_filepath}")

        # 처리 완료된 원본 파일을 archive 폴더로 이동
        archive_filepath = os.path.join(archive_dir, os.path.basename(file_path))
        shutil.move(file_path, archive_filepath)
        logging.info(f"원본 파일 이동 완료: {archive_filepath}")

    except json.JSONDecodeError:
        logging.error(f"JSON 디코딩 오류: {file_path}. 파일을 건너뜁니다.")
        # 문제가 있는 파일은 별도로 처리하거나 이동할 수 있습니다.
    except Exception as e:
        logging.error(f"파일 처리 중 예외 발생 ({file_path}): {e}", exc_info=True)

def main():
    """파이프라인 메인 루프: raw 폴더를 감시하고 분석 작업을 조율합니다."""
    logging.info("===== LLM 분석 파이프라인 시작 =====")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = setup_directories(base_dir)

    agent = get_llm_agent()
    if not agent:
        logging.error("LLM 에이전트 초기화 실패. 파이프라인을 종료합니다.")
        return
        
    analyzer = CafePostAnalyzer(agent)
    logging.info("분석기 초기화 완료. 파일 감시를 시작합니다...")

    try:
        while True:
            # 처리할 파일 목록 찾기 ('_processed_test.json'으로 끝나는 파일만 대상으로 지정)
            files_to_process = [f for f in os.listdir(dirs["raw"]) if f.endswith('_processed_test.json')]
            
            if not files_to_process:
                logging.info(f"'{dirs['raw']}'에 처리할 파일이 없습니다. 60초 후 다시 확인합니다.")
                time.sleep(60)
                continue

            logging.info(f"처리 대기 중인 파일 {len(files_to_process)}개를 발견했습니다.")
            for filename in files_to_process:
                file_path = os.path.join(dirs["raw"], filename)
                process_single_file(file_path, analyzer, dirs["processed"], dirs["archive"])
            
            logging.info("현재 대기열의 모든 파일 처리를 완료했습니다.")

    except KeyboardInterrupt:
        logging.info("사용자에 의해 파이프라인이 중단되었습니다. 프로그램을 종료합니다.")
    except Exception as e:
        logging.error(f"파이프라인 실행 중 심각한 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main() 