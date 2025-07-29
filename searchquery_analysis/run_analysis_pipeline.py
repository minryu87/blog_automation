import os
import time
from datetime import datetime
from core import LabelingSystem
from tqdm import tqdm
import signal
import sys

# --- 우아한 종료를 위한 글로벌 플래그 및 핸들러 ---
keep_running = True
def signal_handler(sig, frame):
    """Ctrl+C 입력을 감지하여 안전한 종료를 준비합니다."""
    global keep_running
    if keep_running:
        print("\n[Ctrl+C 감지] 현재 사이클을 완료한 후 최종 리포트를 저장하고 종료합니다...")
        keep_running = False
    else:
        # 이미 종료 신호를 받은 상태에서 또 누르면 강제 종료
        print("\n[강제 종료] 즉시 프로그램을 종료합니다.")
        sys.exit(1)

# SIGINT (Ctrl+C) 시그널에 대한 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)


# --- Configuration ---
OUTPUT_SUBFOLDER = '0728_1900'  # 결과가 저장될 하위 폴더 이름

# --- Stopping Conditions ---
STOPPING_THRESHOLD = 50  # 미분류 표현 개수가 이 값 미만일 때 중지
MAX_SAFETY_ITERATIONS = 50 # 무한 루프 방지를 위한 최대 반복 횟수
TOP_N_UNLABELED = 50     # 각 반복마다 처리할 미분류 표현의 수

def main():
    print("===== 자동화된 검색어 라벨링 및 택소노미 최적화 시스템 =====")
    
    # --- 1. 시스템 초기화 ---
    print(f"\n[1/4] 시스템 초기화 중...")
    
    # 스크립트 위치를 기준으로 경로를 동적으로 생성
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blog_automation_dir = os.path.dirname(script_dir) # searchquery_analysis의 부모 폴더

    # Assumes 'data/data_input/searchQuery.csv' is within the project structure
    input_file_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'data_input', 'searchQuery.csv'))

    # Construct the output directory path relative to the script directory
    output_dir_base = os.path.abspath(os.path.join(script_dir, 'data', 'data_processed'))
    output_dir_path = os.path.join(output_dir_base, OUTPUT_SUBFOLDER)
    
    print(f"입력 파일: {input_file_path}")
    print(f"결과 저장 폴더: {output_dir_path}")

    # 출력 폴더 생성
    os.makedirs(output_dir_path, exist_ok=True)
    
    system = LabelingSystem(input_file_path=input_file_path, output_dir_path=output_dir_path)
    print("초기화 완료.")

    # --- 2. 반복적인 택소노미 최적화 ---
    print(f"\n[2/4] 택소노미 최적화 반복 작업을 시작합니다 (미분류 표현 < {STOPPING_THRESHOLD} 조건)")
    
    start_time = time.time()

    for cycle in range(1, MAX_SAFETY_ITERATIONS + 1):
        # 루프 시작 전 종료 플래그 확인
        if not keep_running:
            print("종료 신호를 확인하여 메인 루프를 중단합니다.")
            break

        print(f"\n--- Cycle {cycle}/{MAX_SAFETY_ITERATIONS} ---")
        
        system.run_single_labeling_pass()
        
        # 레이블링 후에도 종료 플래그 확인 (긴 작업이므로)
        if not keep_running:
            print("레이블링 작업 후 종료 신호를 확인하여 루프를 중단합니다.")
            break

        unlabeled_df = system.get_unlabeled_report()
        num_unlabeled = len(unlabeled_df)
        print(f"현재 고유 미분류 표현 수: {num_unlabeled}")

        if num_unlabeled == 0:
            print("더 이상 미분류된 표현이 없습니다. 분석을 종료합니다.")
            break
        
        if num_unlabeled <= STOPPING_THRESHOLD:
            print(f"미분류 표현의 수가 {STOPPING_THRESHOLD}개 이하이므로 분석을 종료합니다.")
            break

        print(f"상위 {TOP_N_UNLABELED}개의 미분류 표현을 분석하여 택소노미를 개선합니다...")
        top_unlabeled = unlabeled_df.head(TOP_N_UNLABELED)
        
        updates_made = system.refine_taxonomy(top_unlabeled)
        
        elapsed_time = time.time() - start_time
        print(f"택소노미에 {updates_made}개의 신규 표현이 추가되었습니다.")
        print(f"Cycle {cycle} 완료. (총 소요 시간: {elapsed_time:.2f}초)")


    total_time = time.time() - start_time
    print(f"\n분석이 완료되었거나 중단되었습니다. 총 실행 시간: {total_time:.2f}초")
    
    print("최종 리포트를 생성합니다...")
    system.generate_final_reports()
    print(f"최종 리포트가 다음 경로에 저장되었습니다: {output_dir_path}")

    print("\n\n===== 모든 작업 완료! =====")
    # The original code had a summary print here, but system.generate_final_reports() already prints it.
    # print(summary) 
    print(f"모든 결과는 '{output_dir_path}' 폴더에 저장되었습니다.")
    print("=========================")


if __name__ == "__main__":
    main() 