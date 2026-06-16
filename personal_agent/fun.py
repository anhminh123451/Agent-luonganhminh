"""
Script build ChromaDB từ các file CSV trong data/raw/.

Sử dụng DUY NHẤT class Indexer (facade) với SmartIndexStrategy
để build database. Cách hoạt động:
    1. Copy từng file CSV vào thư mục staging (data/staging/)
    2. Sau mỗi file, chạy Indexer.run() — SmartIndex phát hiện
       data thay đổi và delegate cho IncrementalIndex
    3. Tạm dừng 60 giây sau mỗi file để tránh Gemini API rate limit
    4. Sau khi xong, dọn dẹp thư mục staging

Nhờ SmartIndexStrategy:
    - Lần chạy đầu: phát hiện chưa có hash → full reindex (1 file)
    - Các lần sau: phát hiện hash thay đổi → incremental (chỉ file mới)
    - Tối ưu: không re-embed file đã xử lý

Cách chạy:
    python fun.py
    python fun.py --force          # Force rebuild (xóa index cũ)
    python fun.py --delay 30       # Thay đổi thời gian delay (giây)
    python fun.py --data-dir data/raw  # Thay đổi thư mục source data
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)

# ─── Cấu hình mặc định ───────────────────────────────────────────────
DEFAULT_SOURCE_DIR = "data/raw"
DEFAULT_STAGING_DIR = "data/staging"
DEFAULT_DELAY_SECONDS = 60  # Tạm dừng 60 giây giữa các file


def build_chroma_database(
    source_dir: str = DEFAULT_SOURCE_DIR,
    staging_dir: str = DEFAULT_STAGING_DIR,
    delay_seconds: int = DEFAULT_DELAY_SECONDS,
    force: bool = False,
) -> None:
    """
    Build ChromaDB database từ các file CSV, sử dụng duy nhất Indexer facade.

    Quy trình:
        1. Copy từng file CSV từ source_dir → staging_dir
        2. Chạy Indexer.run() (SmartIndexStrategy) sau mỗi file
        3. Tạm dừng delay_seconds giây giữa các file
        4. Dọn dẹp staging_dir sau khi hoàn tất

    Args:
        source_dir: Thư mục chứa raw data gốc (các file CSV).
        staging_dir: Thư mục staging để Indexer đọc data.
        delay_seconds: Số giây tạm dừng giữa các file.
        force: Nếu True, xóa index cũ và rebuild từ đầu.
    """
    from knowledge_base.indexer import Indexer, SmartIndexStrategy

    source_path = Path(source_dir)
    staging_path = Path(staging_dir)

    # ── Validate thư mục source ──
    if not source_path.exists():
        logger.error(f"Thư mục source không tồn tại: {source_path}")
        sys.exit(1)

    if not source_path.is_dir():
        logger.error(f"Đường dẫn không phải thư mục: {source_path}")
        sys.exit(1)

    # ── Lấy danh sách file CSV ──
    csv_files = sorted([
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".csv"
    ])

    if not csv_files:
        logger.error(f"Không tìm thấy file CSV nào trong {source_path}")
        sys.exit(1)

    total_files = len(csv_files)
    logger.info(f"Tìm thấy {total_files} file CSV trong {source_path}")
    for f in csv_files:
        logger.info(f"  - {f.name} ({f.stat().st_size:,} bytes)")

    # ── Chuẩn bị thư mục staging ──
    staging_path.mkdir(parents=True, exist_ok=True)
    if force:
        # Xóa sạch staging nếu force rebuild
        _clear_directory(staging_path)

    # ── Khởi tạo Indexer (duy nhất facade này) ──
    # Indexer trỏ tới staging_dir — nơi chứa data tích lũy dần
    indexer = Indexer(
        data_dir=staging_dir,
        strategy=SmartIndexStrategy(),
    )

    # ── Force rebuild: xóa toàn bộ index cũ ──
    if force:
        logger.warning("Force rebuild — xóa toàn bộ index cũ...")
        indexer.clear()
        logger.info("Index cũ đã được xóa.")

    # ── Xử lý từng file CSV ──
    start_time = time.time()
    files_processed = []
    files_failed = []
    files_skipped = []

    for idx, csv_file in enumerate(csv_files, start=1):
        filename = csv_file.name
        separator = "─" * 60

        logger.info(f"\n{separator}")
        logger.info(f"[{idx}/{total_files}] Đang xử lý: {filename}")
        logger.info(separator)

        try:
            # ── Step 1: Copy file vào staging ──
            dest = staging_path / filename
            shutil.copy2(csv_file, dest)
            logger.info(f"  📁 Đã copy {filename} → {staging_path}/")

            # ── Step 2: Chạy Indexer (SmartIndex tự detect file mới) ──
            logger.info(f"  🔄 Đang chạy Indexer (SmartIndexStrategy)...")
            result = indexer.run()

            # ── Step 3: Kiểm tra kết quả ──
            if result.skipped:
                logger.info(f"  ⏭ {filename}: Đã được index trước đó — bỏ qua")
                files_skipped.append(filename)
            elif result.success:
                logger.info(
                    f"  ✅ {filename}: Index thành công — "
                    f"{result.total_documents} documents, "
                    f"{result.duration_seconds:.2f}s"
                )
                files_processed.append(filename)
            else:
                logger.error(
                    f"  ❌ {filename}: Index thất bại — {result.error}"
                )
                files_failed.append(filename)

        except Exception as e:
            logger.error(f"  ❌ Lỗi khi xử lý {filename}: {e}")
            files_failed.append(filename)
            # Tiếp tục xử lý file tiếp theo
            continue

        # ── Tạm dừng giữa các file (trừ file cuối) ──
        if idx < total_files and not (result.skipped if 'result' in dir() else False):
            logger.info(
                f"\n  ⏳ Tạm dừng {delay_seconds} giây "
                f"để tránh Gemini API rate limit..."
            )
            _countdown(delay_seconds)

    # ── Báo cáo kết quả cuối cùng ──
    total_duration = time.time() - start_time

    # Lấy trạng thái cuối cùng từ Indexer
    status = indexer.status()

    logger.info(f"\n{'═' * 60}")
    logger.info("  KẾT QUẢ BUILD CHROMA DATABASE")
    logger.info(f"{'═' * 60}")
    logger.info(
        f"  Tổng thời gian      : {total_duration:.2f}s "
        f"({total_duration / 60:.1f} phút)"
    )
    logger.info(f"  File thành công     : {len(files_processed)}/{total_files}")
    logger.info(f"  File đã skip        : {len(files_skipped)}")
    logger.info(f"  File lỗi            : {len(files_failed)}")
    logger.info(f"  Tổng docs trong DB  : {status.document_count}")
    logger.info(f"  Domains             : {status.domains}")
    logger.info(f"  Cần reindex?        : {status.needs_reindex}")
    logger.info(f"{'═' * 60}")

    if files_failed:
        logger.warning(f"  Các file bị lỗi: {files_failed}")

    if files_processed or files_skipped:
        logger.info("  ✅ Build ChromaDB hoàn tất!")
    else:
        logger.error("  ❌ Không có file nào được xử lý thành công.")
        sys.exit(1)


def _clear_directory(dir_path: Path) -> None:
    """Xóa tất cả file trong thư mục (giữ lại thư mục)."""
    if not dir_path.exists():
        return
    for item in dir_path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    logger.debug(f"Đã xóa sạch nội dung thư mục: {dir_path}")


def _countdown(seconds: int) -> None:
    """Hiển thị đếm ngược trên console."""
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        print(f"\r     Còn lại: {mins:02d}:{secs:02d}", end="", flush=True)
        time.sleep(1)
    print("\r     Đã sẵn sàng! Tiếp tục xử lý...       ")


def main():
    """Entry point với CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build ChromaDB từ các file CSV sử dụng Indexer + SmartIndexStrategy, "
            "với rate limit protection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ví dụ:
  python fun.py                       # Chạy mặc định (delay 60s)
  python fun.py --force               # Force rebuild từ đầu
  python fun.py --delay 30            # Delay 30 giây giữa các file
  python fun.py --data-dir data/raw   # Chỉ định thư mục source data
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help=f"Thư mục chứa raw data gốc (mặc định: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default=DEFAULT_STAGING_DIR,
        help=f"Thư mục staging cho Indexer (mặc định: {DEFAULT_STAGING_DIR})",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=DEFAULT_DELAY_SECONDS,
        help=f"Số giây tạm dừng giữa các file (mặc định: {DEFAULT_DELAY_SECONDS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild — xóa index cũ và build lại từ đầu",
    )

    args = parser.parse_args()

    logger.info(
        f"Cấu hình: source_dir={args.data_dir}, staging_dir={args.staging_dir}, "
        f"delay={args.delay}s, force={args.force}"
    )

    build_chroma_database(
        source_dir=args.data_dir,
        staging_dir=args.staging_dir,
        delay_seconds=args.delay,
        force=args.force,
    )


if __name__ == "__main__":
    main()
