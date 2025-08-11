import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger():
    """로거를 설정하고 반환합니다."""
    log_dir = Path(__file__).resolve().parents[2]
    log_file_path = log_dir / 'crawler.log'

    logger = logging.getLogger("performance_analyzer")
    logger.setLevel(logging.DEBUG)  # 로깅 레벨을 DEBUG로 변경

    # 핸들러 중복 추가 방지
    if logger.hasHandlers():
        logger.handlers.clear()

    # 포매터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러
    try:
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError) as e:
        logger.warning(f"로그 파일을 생성할 수 없습니다: {log_file_path}. 에러: {e}")


    return logger

logger = setup_logger()
