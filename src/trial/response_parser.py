"""AIレスポンス文字列からJSONデータを抽出するモジュール。

AIモデルは指示通りにJSONのみを返すとは限らず、```json ``` のコード
ブロックで囲んだり、前後に説明文を付けたりすることがあるため、複数の
抽出パターンを順に試すことでJSONを頑健に取り出す。
"""
import json
import re
from typing import List

_JSON_FENCE_PATTERN = re.compile(
    r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_PATTERN = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)
_BRACKET_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


class ResponseParseError(Exception):
    """AIレスポンスの解析エラー。"""


def extract_json(response_text: str) -> List[dict]:
    """AIのレスポンス文字列からJSON配列を抽出しパースする。

    以下の順序で抽出を試みる。
        1. ```json ... ``` のコードブロック内
        2. ``` ... ``` の任意コードブロック内
        3. レスポンス全体をそのままJSONとして解釈
        4. 最初の '[' から最後の ']' までを抽出

    Args:
        response_text: AIからの生レスポンス文字列。

    Returns:
        パースされたJSON配列(辞書のリスト)。

    Raises:
        ResponseParseError: いずれの方法でもJSONとして解釈できない場合。
    """
    if not response_text or not response_text.strip():
        raise ResponseParseError("レスポンスが空です。")

    candidates: List[str] = []
    candidates.extend(_JSON_FENCE_PATTERN.findall(response_text))
    candidates.extend(_ANY_FENCE_PATTERN.findall(response_text))
    candidates.append(response_text.strip())

    bracket_match = _BRACKET_PATTERN.search(response_text)
    if bracket_match:
        candidates.append(bracket_match.group(0))

    last_error = None
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            return parsed

    raise ResponseParseError(
        f"レスポンスからJSONを抽出できませんでした。詳細: {last_error}")
