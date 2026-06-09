from core.logger import get_logger

logger = get_logger(__name__)

a = int(input("nhap a"))
b = int(input("nhap b"))
c = b-a

if c > 0 :
    logger.debug("hop le voi so tien duong")

if c < 0 :
    logger.warning("cảnh báo bạn đã âm tiền")

if c == 0 :
    logger.info("bạn đã hết tiền")