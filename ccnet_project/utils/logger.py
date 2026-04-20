import logging
from pathlib import Path


def setup_logger(save_dir: str) -> logging.Logger:
    """创建同时输出到终端和文件的日志器。"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ccnet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(save_dir) / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
