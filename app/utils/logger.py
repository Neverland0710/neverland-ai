import logging
from app.config import settings

def setup_logger():
    """로거 설정"""
    
    # 로거 생성
    logger = logging.getLogger("memorial_chat")
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # 이미 핸들러가 있으면 중복 방지
    if logger.handlers:
        return logger
    
    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    
    return logger

# 전역 로거 인스턴스
logger = setup_logger()