from __future__ import annotations

"""
Logging setup cho Banking AI Agent.

Module này cung cấp hệ thống logging chuẩn cho toàn bộ project:
- Tạo logger riêng cho từng module (theo __name__)
- Console handler với màu sắc theo log level
- File handler với RotatingFileHandler (tránh file log quá lớn)
- Log level được điều khiển qua biến môi trường LOG_LEVEL

Cách sử dụng trong các module khác:
    from core.logger import get_logger
    logger = get_logger(__name__)

    logger.debug("Chi tiết debug...")
    logger.info("Thông tin hoạt động...")
    logger.warning("Cảnh báo...")
    logger.error("Lỗi xảy ra...")
    logger.critical("Lỗi nghiêm trọng...")
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from core.config import Settings
setting = Settings()


# ─── Cấu hình mặc định ──────────────────────────────────────────────
DEFAULT_LOG_LEVEL = setting.LOG_LEVEL
DEFAULT_LOG_DIR = setting.LOG_DIR
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Giữ tối đa 3 file log backup


# ─── Màu sắc cho console output (ANSI escape codes) ─────────────────
class _LogColors:
    """ANSI color codes cho từng log level trên terminal."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    DEBUG = "\033[36m"       # Cyan
    INFO = "\033[32m"        # Green
    WARNING = "\033[33m"     # Yellow
    ERROR = "\033[31m"       # Red
    CRITICAL = "\033[1;31m"  # Bold Red


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter thêm màu sắc cho log output trên console.
    Chỉ áp dụng màu khi output là terminal (hỗ trợ ANSI).
    """

    LEVEL_COLORS = {
        logging.DEBUG: _LogColors.DEBUG,
        logging.INFO: _LogColors.INFO,
        logging.WARNING: _LogColors.WARNING,
        logging.ERROR: _LogColors.ERROR,
        logging.CRITICAL: _LogColors.CRITICAL,
    }

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, use_color: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            color = self.LEVEL_COLORS.get(record.levelno, _LogColors.RESET)
            # Tô màu level name
            record.levelname = f"{color}{record.levelname:<8}{_LogColors.RESET}"
            # Tô màu mờ cho timestamp
            record.asctime = f"{_LogColors.DIM}{self.formatTime(record, self.datefmt)}{_LogColors.RESET}"
            # Format message
            formatted = super().format(record)
            return formatted
        return super().format(record)


# ─── Đọc config từ environment / config.py ───────────────────────────
def _get_log_level() -> int:
    """
    Lấy log level từ biến môi trường hoặc config.py.
    Thứ tự ưu tiên: LOG_LEVEL env var > config.py > DEFAULT_LOG_LEVEL
    """
    level_str = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_str, logging.DEBUG)


def _get_log_dir() -> Path:
    """
    Lấy thư mục log. Tự tạo nếu chưa tồn tại.
    Mặc định: <project_root>/logs/
    """
    log_dir_str = os.environ.get("LOG_DIR", DEFAULT_LOG_DIR)
    log_dir = Path(log_dir_str)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# ─── Setup handlers ──────────────────────────────────────────────────
def _create_console_handler(log_level: int) -> logging.StreamHandler:
    """Tạo handler xuất log ra terminal với màu sắc."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Chỉ dùng màu khi output là terminal thực (không phải pipe/file)
    use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    formatter = ColoredFormatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
        use_color=use_color,
    )
    handler.setFormatter(formatter)
    return handler


def _create_file_handler(log_level: int, log_dir: Path) -> RotatingFileHandler:
    """
    Tạo handler ghi log ra file với rotation.
    File log: <log_dir>/agent.log
    Tự động xoay khi file đạt MAX_LOG_FILE_SIZE, giữ BACKUP_COUNT bản backup.
    """
    log_file = log_dir / "agent.log"
    handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=MAX_LOG_FILE_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(log_level)

    # File handler không cần màu sắc
    formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    handler.setFormatter(formatter)
    return handler


# ─── Cấu hình root logger ────────────────────────────────────────────
_initialized = False


def setup_logging() -> None:
    """
    Cấu hình logging cho toàn bộ project.
    Chỉ chạy một lần duy nhất (singleton pattern).

    Gọi hàm này trong entrypoint của app (main.py) để khởi tạo logging.
    Nếu không gọi, get_logger() sẽ tự động gọi khi lần đầu được sử dụng.
    """
    global _initialized
    if _initialized:
        return

    log_level = _get_log_level()
    log_dir = _get_log_dir()

    # Cấu hình root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Xóa handlers cũ (tránh duplicate khi gọi lại)
    root_logger.handlers.clear()

    # Thêm handlers
    root_logger.addHandler(_create_console_handler(log_level))
    root_logger.addHandler(_create_file_handler(log_level, log_dir))

    # Giảm noise từ thư viện bên thứ ba
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)

    _initialized = True

    # Log thong tin khoi tao
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Logging initialized successfully")
    logger.info(f"  Log level : {logging.getLevelName(log_level)}")
    logger.info(f"  Log file  : {log_dir / 'agent.log'}")
    logger.info("=" * 60)


# ─── Factory function (API chính) ────────────────────────────────────
def get_logger(name: str | None = None) -> logging.Logger:
    """
    Tạo hoặc lấy logger cho một module cụ thể.

    Args:
        name: Tên module, thường dùng __name__ để tự lấy tên module.
              Ví dụ: get_logger(__name__) trong file core/config.py
              sẽ tạo logger có tên "core.config"

    Returns:
        logging.Logger: Logger đã được cấu hình sẵn.

    Ví dụ sử dụng:
        from core.logger import get_logger
        logger = get_logger(__name__)

        logger.info("Server đang khởi động...")
        logger.error("Không thể kết nối database", exc_info=True)
    """
    # Tự động setup nếu chưa khởi tạo
    if not _initialized:
        setup_logging()

    return logging.getLogger(name)
