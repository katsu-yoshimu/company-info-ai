"""
Company Intelligence Extractor CLI

AI (Claude, ChatGPT, Gemini) を利用して会社名・住所・電話番号から
公式ホームページURL、メールアドレス、SNSアカウント情報を取得・抽出するスクリプト。
Claude/ChatGPT利用時はDuckDuckGo検索を前処理として実行し、最新情報を提供します。
"""

import abc
import argparse
import configparser
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List


# ==========================================
# Web Search Utility (DuckDuckGo)
# ==========================================

def search_duckduckgo(query: str, max_results: int = 6) -> str:
    """
    DuckDuckGoを使って指定クエリでWeb検索を行います。
    SNSアカウントの取りこぼしを防ぐため max_results を 6 件に拡張。
    """
    import time
    time.sleep(1.5)  # 連続アクセス制限を回避するための待機時間
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region="jp-jp", max_results=max_results))
            if not results:
                return "検索結果なし"

            snippets = []
            for item in results:
                title = item.get("title", "")
                url = item.get("href", "")
                body = item.get("body", "")
                snippets.append(f"- タイトル: {title}\n  URL: {url}\n  概要: {body}")

            return "\n".join(snippets)
    except Exception as err:
        return f"（検索処理中にエラーが発生しました: {err}）"


# ==========================================
# Design Pattern: Strategy Pattern for LLMs
# ==========================================

class BaseLLMClient(abc.ABC):
    """LLMプロバイダ向けの抽象基底クラス (Strategy Pattern)"""

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
    """Google Gemini API用クライアント (Google検索機能付き)"""

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
# Design Pattern: Factory Pattern for LLMs
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
# Utility & Main Functions
# ==========================================

def load_tsv_file(file_path: str) -> List[Dict[str, str]]:
    """
    指定されたTAB形式(TSV)ファイルを読み込んで辞書リストとして返します。
    列の欠損やNone値が含まれる場合も安全にハンドリングします。

    Args:
        file_path (str): TSVファイルのパス

    Returns:
        List[Dict[str, str]]: 会社情報(会社名, 住所, 電話番号)のリスト

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: ヘッダーに必須フィールドが存在しない場合
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"指定ファイルが存在しません: {file_path}")

    companies = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required_keys = {"company_name", "address", "phone"}

        if (
            reader.fieldnames is None
            or not required_keys.issubset(set(reader.fieldnames))
        ):
            raise ValueError(
                f"TSVヘッダーには {required_keys} が必要です。"
            )

        for line_num, row in enumerate(reader, start=2):
            company_name = (row.get("company_name") or "").strip()
            address = (row.get("address") or "").strip()
            phone = (row.get("phone") or "").strip()

            if not company_name:
                sys.stderr.write(
                    f"[WARN] {line_num}行目の会社名を取得できませんでした (スキップします)。\n"
                )
                continue

            companies.append({
                "company_name": company_name,
                "address": address,
                "phone": phone
            })

    return companies


def build_prompt(
    template: str,
    companies_batch: List[Dict[str, str]],
    provider: str
) -> str:
    """
    プロンプトテンプレートに会社データおよびWeb検索結果を組み込みます。
    通常検索に加え、SNS特化検索を実施して取りこぼしを防止します。
    """
    input_data = []
    perform_web_search = provider in ["claude", "chatgpt", "openai"]

    for company in companies_batch:
        item = dict(company)

        if perform_web_search:
            c_name = company.get("company_name", "")
            c_addr = company.get("address", "")
            
            # ① 基本検索（HP・メール・全体情報）
            main_query = f"{c_name} {c_addr} 公式ホームページ メール"
            main_snippet = search_duckduckgo(main_query, max_results=5)
            
            # ② SNS特化検索（Instagram / Facebook / LINE を明記してピンポイント検索）
            sns_query = f"{c_name} {c_addr} instagram facebook line"
            sns_snippet = search_duckduckgo(sns_query, max_results=5)

            # 両方の検索結果を合成してLLMに渡す
            item["web_search_results"] = (
                f"--- [基本検索結果] ---\n{main_snippet}\n\n"
                f"--- [SNS特化検索結果] ---\n{sns_snippet}"
            )

        input_data.append(item)

    input_json_str = json.dumps(input_data, ensure_ascii=False, indent=2)
    return template.replace("{companies_data}", input_json_str)


def parse_json_response(
    raw_text: str, expected_batch: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
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
        sys.stderr.write(
            f"[WARN] JSONパース失敗。フォールバックデータ(null)を適用します。"
            f" エラー: {err}\n"
        )

    # パースに成功し、正常なリストが得られた場合
    if isinstance(parsed_data, list) and len(parsed_data) > 0:
        return parsed_data

    # パース失敗時・空文字応答時のフォールバック処理（全項目 null で補填）
    fallback_results = []
    for company in expected_batch:
        fallback_results.append({
            "company_name": company.get("company_name", "不明"),
            "official_website": None,
            "official_email": None,
            "instagram": None,
            "facebook": None,
            "line": None
        })
    return fallback_results


def print_results(results: List[Dict[str, Any]]) -> None:
    """
    抽出した会社情報を標準出力に表示します。

    Args:
        results (List[Dict[str, Any]]): 解析済み会社情報リスト
    """
    for idx, item in enumerate(results, start=1):
        print("=" * 65)
        print(f"【件数 #{idx}】 会社名: {item.get('company_name', '不明')}")
        print("-" * 65)
        print(f"  公式HP URL            : {item.get('official_website') or '未取得'}")
        print(f"  公式メールアドレス     : {item.get('official_email') or '未取得'}")
        print(f"  Instagram アカウント  : {item.get('instagram') or '未取得'}")
        print(f"  Facebook アカウント   : {item.get('facebook') or '未取得'}")
        print(f"  LINE アカウント       : {item.get('line') or '未取得'}")
    print("=" * 65)


def main() -> None:
    """
    メイン実行フロー。
    引数パース、設定読み込み、TSV処理、AI呼び出し、結果表示を行います。
    """
    parser = argparse.ArgumentParser(
        description="会社情報(TSV)からAIを利用してWeb・SNS情報を抽出するツール"
    )
    parser.add_argument(
        "tsv_file",
        type=str,
        help="入力するTAB形式(TSV)ファイルのパス"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_v1.ini",
        help="設定ファイルのパス (デフォルト: config.ini)"
    )
    args = parser.parse_args()

    # 設定ファイルの読み込み
    if not os.path.exists(args.config):
        print(f"エラー: 設定ファイル '{args.config}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(args.config, encoding="utf-8")

    try:
        batch_size = config.getint("SETTINGS", "batch_size", fallback=5)
        prompt_template = config.get("PROMPT", "template")
        provider = config.get("LLM", "provider").lower()
        llm_client = LLMFactory.create_client(config)
    except Exception as err:
        print(f"設定ファイル読み込みエラー: {err}", file=sys.stderr)
        sys.exit(1)

    # TSVファイルの読み込み
    try:
        companies = load_tsv_file(args.tsv_file)
    except Exception as err:
        print(f"TSVファイル読み込みエラー: {err}", file=sys.stderr)
        sys.exit(1)

    if not companies:
        print("処理対象のデータが存在しません。")
        sys.exit(0)

    # バッチ処理の実行
    all_results = []
    total_companies = len(companies)

    for i in range(0, total_companies, batch_size):
        batch = companies[i: i + batch_size]
        print(batch)
        prompt = build_prompt(prompt_template, batch, provider)

        try:
            raw_response = llm_client.generate_text(prompt)
            parsed_batch_results = parse_json_response(raw_response, batch)
            all_results.extend(parsed_batch_results)
        except Exception as err:
            print(
                f"AI処理中にエラーが発生しました (バッチ {i // batch_size + 1}): {err}",
                file=sys.stderr
            )
            # APIエラー発生時も処理を途中で止めず、nullデータを補填して続行
            fallback = [
                {
                    "company_name": c.get("company_name", "不明"),
                    "official_website": None,
                    "official_email": None,
                    "instagram": None,
                    "facebook": None,
                    "line": None
                }
                for c in batch
            ]
            all_results.extend(fallback)

    # 標準出力へ表示
    print_results(all_results)


if __name__ == "__main__":
    main()