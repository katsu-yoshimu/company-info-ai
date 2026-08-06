"""会社情報CSVの読み込み・バッチ分割モジュール。"""
import csv
from typing import Dict, Iterator, List

REQUIRED_COLUMNS = ["company_name", "address", "phone"]


class CSVFormatError(Exception):
    """CSVファイルの形式エラー。"""


def read_companies(csv_path: str) -> List[Dict[str, str]]:
    """会社情報CSVファイル(company_name, address, phone列)を読み込む。

    Args:
        csv_path: CSVファイルのパス。

    Returns:
        会社名、住所、電話番号を保持する辞書のリスト。

    Raises:
        CSVFormatError: ヘッダーがない、または必須列が不足している場合。
        FileNotFoundError: CSVファイルが存在しない場合。
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise CSVFormatError("CSVファイルにヘッダー行がありません。")

        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise CSVFormatError(f"CSVファイルに必須列がありません: {missing}")

        companies = []
        for row_num, row in enumerate(reader, start=2):
            company_name = (row.get("company_name") or "").strip()
            address = (row.get("address") or "").strip()
            phone = (row.get("phone") or "").strip()

            if not company_name:
                raise CSVFormatError(
                    f"{row_num}行目: company_name が空です。")

            companies.append({
                "company_name": company_name,
                "address": address,
                "phone": phone,
            })

    return companies


def batch_companies(
        companies: List[Dict[str, str]],
        batch_size: int) -> Iterator[List[Dict[str, str]]]:
    """会社情報リストを指定件数ごとに分割する。

    Args:
        companies: 会社情報の辞書リスト。
        batch_size: 1バッチあたりの件数(config.yaml の batch.size)。

    Yields:
        分割された会社情報のリスト。

    Raises:
        ValueError: batch_size が1未満の場合。
    """
    if batch_size <= 0:
        raise ValueError("batch_size は1以上を指定してください。")

    for i in range(0, len(companies), batch_size):
        yield companies[i:i + batch_size]
