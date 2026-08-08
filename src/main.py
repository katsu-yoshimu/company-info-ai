"""
Company Intelligence Extractor CLI

AI (Claude, ChatGPT, Gemini) を利用して会社名・住所・電話番号から
公式ホームページURL、メールアドレス、SNSアカウント情報を取得・抽出するスクリプト。

変更点:
- 入力ファイルを CSV に変更 (DictReader で標準のカンマ区切り解析)
- 日付入りログファイル名対応 (config.ini のテンプレート設定 + {date} 置換)
- ログレベルの個別設定対応 (標準出力 / ファイル出力 それぞれのレベルを config から取得)
"""

import abc
import argparse
import configparser
import csv
import json
import logging
import os
import re
import sys
import time
import datetime
import traceback
from typing import Any, Dict, List


# ==========================================
# 1. Logger Setup
# ==========================================

def setup_logger(
    log_file_template: str = "process_{date}.log",
    console_level_str: str = "INFO",
    file_level_str: str = "DEBUG"
) -> logging.Logger:
    """
    標準出力(sys.stdout)とファイルへ同時にログを出力するロガーを設定します。
    日付入りファイル名および出力レベルを引数で制御します。
    """
    logger = logging.getLogger("company_processor")
    
    # ロガー自体のルートレベルは最も低い(詳細な)方に合わせる
    console_level = getattr(logging, console_level_str.upper(), logging.INFO)
    file_level = getattr(logging, file_level_str.upper(), logging.DEBUG)
    logger.setLevel(min(console_level, file_level))

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 日付文字列 (YYYYMMDD) の生成とファイル名の組み立て
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    log_file = log_file_template.format(date=today_str)

    # ① コンソール（標準出力）用ハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ② ファイル出力用ハンドラ（UTF-8）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ==========================================
# 2. Web Search Utility (ddgs / DuckDuckGo)
# ==========================================

def search_duckduckgo(query: str, max_results: int = 5) -> str:
    """
    DuckDuckGoを使って指定クエリでWeb検索を行い、テキスト形式で返します。
    レートリミット（429エラー）回避のため呼び出し前に1.5秒待機します。
    """
    time.sleep(1.5)  # レートリミット回避のためのウェイト
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="jp-jp", max_results=max_results))
            if not results:
                return "（該当する検索結果が見つかりませんでした）"

            snippets = []
            for item in results:
                title = item.get("title", "")
                url = item.get("href", "")
                body = item.get("body", "")
                snippets.append(f"- タイトル: {title}\n  URL: {url}\n  概要: {body}")

            return "\n".join(snippets)
    except ImportError:
        logging.getLogger("company_processor").warning(
            "ddgs パッケージが未インストールです。"
            " Web検索を行わずにプロンプトを生成します。(pip install ddgs を推奨)"
        )
        return "（Web検索ライブラリ未利用）"
    except Exception as err:
        logging.getLogger("company_processor").warning(f"DuckDuckGo検索処理中にエラーが発生しました: {err}")
        return f"（検索処理エラー: {err}）"


# ==========================================
# 3. Design Pattern: Strategy Pattern for LLMs
# ==========================================

class BaseLLMClient(abc.ABC):
    """LLMプロバイダ向けの抽象基底クラス"""

    @abc.abstractmethod
    def generate_text(self, prompt: str) -> str:
        """
        指定されたプロンプトに基づいてテキストを生成します。

        Args:
            prompt (str): LLMに送るプロンプト文字列

        Returns:
            str: LLMからの応答文字列
        """
        pass


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude API用クライアント"""

    def __init__(self, api_key: str, model: str):
        """
        ClaudeClientの初期化処理。

        Args:
            api_key (str): Anthropic APIキー
            model (str): 使用するモデル名
        """
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError(
                "anthropic パッケージが未インストールです。"
                "'pip install anthropic' を実行してください。"
            )

    def generate_text(self, prompt: str) -> str:
        """
        Claude APIを実行してレスポンス文字列を取得します。
        ThinkingBlock等の非テキストブロックが含まれる場合も安全にテキストのみを抽出します。

        Args:
            prompt (str): プロンプト文字列

        Returns:
            str: 生成されたテキスト
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        # contentブロックの中からテキスト要素のみを抽出して結合
        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text" or hasattr(block, "text"):
                text_parts.append(block.text)

        return "\n".join(text_parts)


class OpenAIClient(BaseLLMClient):
    """OpenAI ChatGPT API用クライアント"""

    def __init__(self, api_key: str, model: str):
        """
        OpenAIClientの初期化処理。

        Args:
            api_key (str): OpenAI APIキー
            model (str): 使用するモデル名
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError(
                "openai パッケージが未インストールです。"
                "'pip install openai' を実行してください。"
            )

    def generate_text(self, prompt: str) -> str:
        """
        OpenAI APIを実行してレスポンス文字列を取得します。

        Args:
            prompt (str): プロンプト文字列

        Returns:
            str: 生成されたテキスト
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content or ""


class GeminiClient(BaseLLMClient):
    """Google Gemini API用クライアント (Google Grounding付き)"""

    def __init__(self, api_key: str, model: str):
        """
        GeminiClientの初期化処理。

        Args:
            api_key (str): Google AI APIキー
            model (str): 使用するモデル名
        """
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError(
                "google-genai パッケージが未インストールです。"
                "'pip install google-genai' を実行してください。"
            )

    def generate_text(self, prompt: str) -> str:
        """
        Gemini APIを実行してレスポンス文字列を取得します。
        リアルタイムのGoogle検索機能(Grounding)を有効化してWeb情報を検索します。

        Args:
            prompt (str): プロンプト文字列

        Returns:
            str: 生成されたテキスト
        """
        from google.genai import types

        # Google検索ツール(Grounding)を有効化
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text or ""


# ==========================================
# 4. Design Pattern: Factory Pattern for LLMs
# ==========================================

class LLMFactory:
    """LLMクライアントインスタンスを生成するファクトリクラス"""

    @staticmethod
    def create_client(config: configparser.ConfigParser) -> BaseLLMClient:
        """
        設定ファイルの内容に基づき適切なLLMクライアントを生成します。

        Args:
            config (configparser.ConfigParser): 読み込み済みの設定オブジェクト

        Returns:
            BaseLLMClient: 生成されたLLMクライアントのインスタンス

        Raises:
            ValueError: サポート対象外のプロバイダが指定された場合
        """
        provider = config.get("LLM", "provider").lower()
        api_key = config.get("LLM", "api_key")
        model = config.get("LLM", "model")

        if provider == "claude":
            return ClaudeClient(api_key=api_key, model=model)
        elif provider in ["chatgpt", "openai"]:
            return OpenAIClient(api_key=api_key, model=model)
        elif provider == "gemini":
            return GeminiClient(api_key=api_key, model=model)
        else:
            raise ValueError(f"未対応のLLMプロバイダです: {provider}")


# ==========================================
# 5. Utility & Processing Functions
# ==========================================

def load_csv_file(file_path: str, logger: logging.Logger) -> List[Dict[str, str]]:
    """
    指定されたCSV形式(CSV)ファイルを読み込んで辞書リストとして返します。
    列の欠損やNone値が含まれる場合も安全にハンドリングします。

    Args:
        file_path (str): CSVファイルのパス
        logger (logging.Logger): logger

    Returns:
        List[Dict[str, str]]: 会社情報(会社名, 住所, 電話番号)のリスト

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: ヘッダーに必須フィールドが存在しない場合
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"指定ファイルが存在しません: {file_path}")

    companies = []
    # UTF-8 (BOM付き含む) に対応するために utf-8-sig を使用
    with open(file_path, mode="r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)  # デフォルトでカンマ区切り (CSV)
        required_keys = {"company_name", "address", "phone"}

        if reader.fieldnames is None or not required_keys.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSVヘッダーには {required_keys} が含まれている必要があります。")

        for line_num, row in enumerate(reader, start=2):
            company_name = (row.get("company_name") or "").strip()
            address = (row.get("address") or "").strip()
            phone = (row.get("phone") or "").strip()

            if not company_name:
                logger.warning(f"{line_num}行目の会社名を取得できませんでした (スキップします)。")
                continue

            companies.append({
                "company_name": company_name,
                "address": address,
                "phone": phone
            })

    return companies


def build_prompt(
    template: str,
    company: Dict[str, str],
    provider: str
) -> str:
    """
    1社分のデータを基に送信用プロンプトを作成します。
    Claude/ChatGPTの場合は基本検索とSNS特化検索の結果を追加します。
    """
    item = dict(company)

    if provider in ["claude", "chatgpt", "openai"]:
        c_name = company.get("company_name", "")
        c_addr = company.get("address", "")

        main_query = f"{c_name} {c_addr} 公式ホームページ メール"
        main_snippet = search_duckduckgo(main_query, max_results=5)

        sns_query = f"{c_name} {c_addr} instagram facebook line"
        sns_snippet = search_duckduckgo(sns_query, max_results=5)

        item["web_search_results"] = (
            f"--- [基本検索結果] ---\n{main_snippet}\n\n"
            f"--- [SNS特化検索結果] ---\n{sns_snippet}"
        )

    input_json_str = json.dumps([item], ensure_ascii=False, indent=2)
    return template.replace("{companies_data}", input_json_str)


def parse_json_response(
    raw_text: str, target_company: Dict[str, str], logger: logging.Logger
) -> Dict[str, Any]:
    """
    LLMからの応答文字列からJSON部分を抽出してパースします。
    パース失敗や件数不足の場合、入力バッチに対応する null 埋めデータを補填します。

    Args:
        raw_text (str): LLMからの生のレスポンス文字列
        expected_batch (List[Dict[str, str]]): 送信した元の会社情報バッチ

    Returns:
        List[Dict[str, Any]]: パース済み（またはフォールバック補填済み）の辞書リスト
    """
    cleaned_text = raw_text.strip()
    parsed_data = None

    # Markdownのコードブロック記号 (```json ... ```) を除去
    if "```" in cleaned_text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_text)
        if match:
            cleaned_text = match.group(1).strip()

    # JSON配列 [...] の部分を正規表現で抽出
    json_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", cleaned_text)
    if json_match:
        cleaned_text = json_match.group(0)

    try:
        if cleaned_text:
            parsed_data = json.loads(cleaned_text)
    except json.JSONDecodeError as err:
        logger.warning(f"JSONパースに失敗しました: {err}")

    if isinstance(parsed_data, list) and len(parsed_data) > 0:
        return parsed_data[0]
    elif isinstance(parsed_data, dict):
        return parsed_data

    return {
        "company_name": target_company.get("company_name", "不明"),
        "official_website": None,
        "official_email": None,
        "instagram": None,
        "facebook": None,
        "line": None
    }


def print_results(results: List[Dict[str, Any]], logger) -> None:
    """
    抽出した会社情報を標準出力に表示します。

    Args:
        results (List[Dict[str, Any]]): 解析済み会社情報リスト
    """
    for idx, item in enumerate(results, start=1):
        logger.info("-" * 65)
        logger.info(f"【件数 #{idx}】 会社名: {item.get('company_name', '不明')}")
        logger.info("-" * 65)
        logger.info(f"  公式HP URL            : {item.get('official_website') or '未取得'}")
        logger.info(f"  公式メールアドレス     : {item.get('official_email') or '未取得'}")
        logger.info(f"  Instagram アカウント  : {item.get('instagram') or '未取得'}")
        logger.info(f"  Facebook アカウント   : {item.get('facebook') or '未取得'}")
        logger.info(f"  LINE アカウント       : {item.get('line') or '未取得'}")
    logger.info("=" * 65)


# ==========================================
# 6. Main Execution Flow
# ==========================================

def main() -> None:
    """
    メイン実行フロー。
    引数パース、設定読み込み、TSV処理、AI呼び出し、結果表示を行います。
    """
    parser = argparse.ArgumentParser(
        description="会社情報(CSV)からAIを利用してWeb・SNS情報を抽出するツール"
    )
    parser.add_argument("--input", type=str, default="input/companies.csv", help="入力するCSVファイルのパス")
    parser.add_argument("--config", type=str, default="config.ini", help="設定ファイルパス")
    args = parser.parse_args()

    # 設定ファイルの読み込み
    if not os.path.exists(args.config):
        print(f"[ERROR] 設定ファイル '{args.config}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(args.config, encoding="utf-8")

    # ロガー設定を config.ini より取得
    log_file_template = config.get("LOG", "log_file_template", fallback="logs/process_{date}.log")
    console_level_str = config.get("LOG", "console_level", fallback="INFO")
    file_level_str = config.get("LOG", "file_level", fallback="DEBUG")

    logger = setup_logger(
        log_file_template=log_file_template,
        console_level_str=console_level_str,
        file_level_str=file_level_str
    )

    try:
        prompt_template = config.get("PROMPT", "template")
        provider = config.get("LLM", "provider").lower()
        model_name = config.get("LLM", "model")
        llm_client = LLMFactory.create_client(config)
    except Exception as err:
        logger.error(f"設定読み込みエラー: {err}\n{traceback.format_exc()}")
        sys.exit(1)

    # CSVファイルの読み込み
    try:
        companies = load_csv_file(args.input, logger)
    except Exception as err:
        logger.error(f"CSV読み込みエラー: {err}\n{traceback.format_exc()}")
        sys.exit(1)

    if not companies:
        logger.info("処理対象のデータが存在しません。")
        sys.exit(0)

    total_companies = len(companies)

    # 開始ログ出力
    logger.info("====================================================")
    logger.info(f"=== 会社情報抽出処理を開始します (対象: 全 {total_companies} 件) ===")
    logger.info(f"使用AIプロバイダ : {provider}")
    logger.info(f"使用モデル       : {model_name}")
    logger.info("====================================================")

    all_results = []

    for idx, company in enumerate(companies, start=1):
        c_name = company.get("company_name", "名称不明")

        logger.info("----------------------------------------------------")
        logger.info(f"進捗 [{idx}/{total_companies}] 処理開始: {c_name}")

        try:
            # 1. プロンプト生成
            prompt = build_prompt(prompt_template, company, provider)

            # 2. リクエストログ (ファイル側が DEBUG などで設定されている場合に詳細記録)
            logger.debug(f"[{c_name}] --- AI Request Prompt ---\n{prompt}")

            # 3. AI API呼出
            raw_response = llm_client.generate_text(prompt)

            # 4. レスポンスログ
            logger.debug(f"[{c_name}] --- AI Raw Response ---\n{raw_response}")

            # 5. 結果のパース
            parsed_result = parse_json_response(raw_response, company, logger)
            all_results.append(parsed_result)

            logger.info(f"進捗 [{idx}/{total_companies}] 処理成功: {c_name}")

        except Exception as err:
            logger.error(
                f"進捗 [{idx}/{total_companies}] 処理失敗: {c_name} | エラー: {err}\n"
                f"スタックトレース:\n{traceback.format_exc()}"
            )
            fallback = {
                "company_name": c_name,
                "official_website": None,
                "official_email": None,
                "instagram": None,
                "facebook": None,
                "line": None
            }
            all_results.append(fallback)

    logger.info("----------------------------------------------------")
    logger.info(f"=== 全 {total_companies} 件の処理が完了しました ===")

    # 結果をコンソールに表示
    print_results(all_results, logger)

    # 処理終了後にウィンドウが即座に閉じないようEnterキー入力を待つ
    input("処理が完了しました。[Enter] キーを押して終了してください...")


if __name__ == "__main__":
    main()