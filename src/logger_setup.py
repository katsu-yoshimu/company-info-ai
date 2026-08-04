"""ロギング設定モジュール。

標準出力とログファイルの両方にログを出力できるロガーを提供する。
AIへのプロンプト送信内容・レスポンス内容はDEBUGレベルで記録する。
"""
import logging
import os


def setup_logger(
        log_file: str, console_level: str = "INFO",
        file_level: str = "DEBUG") -> logging.Logger:
    """アプリケーション全体で利用するロガーを設定する。

    Args:
        log_file: ログファイルの出力パス。
        console_level: コンソール出力のログレベル(INFO等)。
        file_level: ファイル出力のログレベル(DEBUG等)。

    Returns:
        設定済みのロガーインスタンス。
    """
    logger = logging.getLogger("company_info_extractor")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        getattr(logging, console_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(
        getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
