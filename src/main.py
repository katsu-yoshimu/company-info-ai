#!/usr/bin/env python3
"""会社情報(会社名/住所/電話番号)から、公式ホームページURL・公式メール

アドレス・Instagram/Facebook/LINE公式アカウントをAIで調査するスクリプト。

使用例:
    python main.py companies.csv --config config.yaml
"""
import argparse
import logging
import sys
from typing import Dict, List

from ai_providers import AIProviderBase, AIProviderError, AIProviderFactory
from config_loader import ConfigError, load_config
from csv_handler import CSVFormatError, batch_companies, read_companies
from logger_setup import setup_logger
from prompt_builder import build_followup_prompt, build_prompt
from response_parser import ResponseParseError, extract_json

RESULT_FIELDS = ["website_url", "email", "instagram", "facebook", "line"]

# 初回調査で見送られがちな項目。フォローアップ(再調査)の対象とする。
FOLLOWUP_FIELDS = ["email", "instagram", "facebook", "line"]

RESULT_LABELS = {
    "website_url": "公式ホームページURL",
    "email": "公式メールアドレス",
    "instagram": "Instagram公式アカウント",
    "facebook": "Facebook公式アカウント",
    "line": "LINE公式アカウント",
}


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Args:
        argv: 解析対象の引数リスト。Noneの場合は sys.argv を使用する。

    Returns:
        解析済みの引数(csv_file, config)。
    """
    parser = argparse.ArgumentParser(
        description="会社情報からAIを用いて公式SNS/連絡先情報を取得する。")
    parser.add_argument(
        "csv_file",
        help="会社名(company_name)/住所(address)/電話番号(phone)列を"
             "含むCSVファイルのパス")
    parser.add_argument(
        "--config", default="config.yaml",
        help="設定ファイルのパス(デフォルト: config.yaml)")
    return parser.parse_args(argv)


def match_result(company: Dict[str, str], results: List[dict]) -> dict:
    """AIレスポンスの結果一覧から対象会社に対応する結果を検索する。

    会社名の完全一致を優先し、見つからない場合は部分一致で検索する。

    Args:
        company: 検索対象の会社情報(company_name/address/phone)。
        results: AIレスポンスから抽出した結果の辞書リスト。

    Returns:
        該当する結果の辞書。見つからない場合は空の辞書。
    """
    target_name = company["company_name"]

    for result in results:
        if isinstance(result, dict) and result.get("company_name") == \
                target_name:
            return result

    for result in results:
        if not isinstance(result, dict):
            continue
        result_name = result.get("company_name") or ""
        if target_name and result_name and (
                target_name in result_name or result_name in target_name):
            return result

    return {}


def get_missing_field_labels(result: dict) -> List[str]:
    """結果からFOLLOWUP_FIELDSのうち未確認(null/空)の項目ラベルを返す。

    Args:
        result: match_result で得られた1社分の結果辞書。

    Returns:
        未確認項目の日本語ラベルのリスト。全て確認済みの場合は空リスト。
    """
    return [
        RESULT_LABELS[field] for field in FOLLOWUP_FIELDS
        if not result.get(field)
    ]


def refine_missing_fields(
        company: Dict[str, str], result: dict,
        provider: AIProviderBase, followup_config: dict,
        logger: logging.Logger) -> dict:
    """未確認項目がある1社について、より粘り強く再調査する。

    初回調査では「表示テキストにURLが見えない」等の理由で保守的に
    nullとされたケースを、より具体的な指示(リンク先URLの確認、
    他ページの確認、グループ名検索、表記ゆれ再検索)を与えて拾い直す。
    再調査でも確認できなかった項目はnullのまま維持する。

    Args:
        company: 対象の会社情報(company_name/address/phone)。
        result: 初回調査の結果辞書(match_resultの戻り値)。
        provider: 利用するAIProviderインスタンス。
        followup_config: config.yaml の followup セクション。
        logger: ロガーインスタンス。

    Returns:
        再調査結果をマージした結果辞書。再調査が無効・不要な場合は
        引数の result をそのまま返す。
    """
    if not followup_config.get("enabled"):
        return result

    missing_labels = get_missing_field_labels(result)
    if not missing_labels:
        return result

    template = followup_config.get("template")
    if not template:
        logger.warning(
            "followup.enabled が true ですが followup.template が"
            "設定されていないため、再調査をスキップします。")
        return result

    logger.info(
        "「%s」の未確認項目(%s)について再調査します。",
        company["company_name"], "、".join(missing_labels))

    prompt = build_followup_prompt(template, company, missing_labels)

    try:
        response_text = provider.generate(prompt)
        followup_results = extract_json(response_text)
    except (AIProviderError, ResponseParseError) as exc:
        logger.warning(
            "「%s」の再調査に失敗したため、初回結果を維持します: %s",
            company["company_name"], exc)
        return result

    if not followup_results:
        return result

    followup_result = followup_results[0]
    merged = dict(result)
    for field in FOLLOWUP_FIELDS:
        new_value = followup_result.get(field)
        if new_value and not merged.get(field):
            merged[field] = new_value
            logger.info(
                "「%s」の%sを再調査で確認できました: %s",
                company["company_name"], RESULT_LABELS[field], new_value)

    return merged


def print_result(company: Dict[str, str], result: dict) -> None:
    """1件分の結果を標準出力に表示する。

    情報が取得できなかった項目は「取得できませんでした」と表示する。

    Args:
        company: 対象の会社情報(company_name/address/phone)。
        result: AIから取得した結果情報。取得できない場合は空辞書。
    """
    print("-" * 60)
    print(f"会社名   : {company['company_name']}")
    print(f"住所     : {company['address']}")
    print(f"電話番号 : {company['phone']}")

    if not result:
        print("結果     : 情報を取得できませんでした。")
        return

    for field in RESULT_FIELDS:
        value = result.get(field)
        display_value = value if value else "取得できませんでした"
        print(f"{RESULT_LABELS[field]:<22}: {display_value}")


def process_batch(
        batch_index: int, total_batches: int,
        companies: List[Dict[str, str]],
        provider: AIProviderBase, template: str,
        followup_config: dict,
        logger: logging.Logger) -> None:
    """1バッチ分の会社情報をAIに問い合わせ、結果を出力する。

    バッチ単位でAI呼び出し失敗・JSON解析失敗が発生しても、当該バッチの
    各社について「取得できませんでした」として結果を表示し、処理全体は
    継続する。

    Args:
        batch_index: 現在処理中のバッチ番号(1始まり)。
        total_batches: 全バッチ数。
        companies: 当該バッチの会社情報リスト。
        provider: 利用するAIProviderインスタンス。
        template: プロンプトのテンプレート文字列。
        followup_config: config.yaml の followup セクション
            (未確認項目の再調査設定)。
        logger: ロガーインスタンス。
    """
    print(f"\n=== バッチ {batch_index}/{total_batches} "
          f"({len(companies)}件) を処理中... ===")
    logger.info("バッチ %d/%d の処理を開始します(%d件)。",
                batch_index, total_batches, len(companies))

    prompt = build_prompt(template, companies)

    try:
        response_text = provider.generate(prompt)
    except AIProviderError as exc:
        logger.error("AI呼び出しに失敗しました: %s", exc)
        print("AI呼び出しに失敗したため、このバッチはスキップします。")
        for company in companies:
            print_result(company, {})
        return

    try:
        results = extract_json(response_text)
    except ResponseParseError as exc:
        logger.error("レスポンスのJSON解析に失敗しました: %s", exc)
        print("レスポンスの解析に失敗したため、このバッチの情報は"
              "取得できませんでした。")
        for company in companies:
            print_result(company, {})
        return

    for company in companies:
        result = match_result(company, results)
        result = refine_missing_fields(
            company, result, provider, followup_config, logger)
        print_result(company, result)

    logger.info("バッチ %d/%d の処理が完了しました。",
                batch_index, total_batches)


def main(argv: List[str] = None) -> int:
    """スクリプトのエントリポイント。

    Args:
        argv: コマンドライン引数リスト(テスト用)。Noneの場合はsys.argv。

    Returns:
        終了コード(正常終了時は0、異常終了時は1)。
    """
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"設定ファイルの読み込みに失敗しました: {exc}", file=sys.stderr)
        return 1

    logging_conf = config["logging"]
    logger = setup_logger(
        log_file=logging_conf.get("log_file", "logs/app.log"),
        console_level=logging_conf.get("console_level", "INFO"),
        file_level=logging_conf.get("file_level", "DEBUG"),
    )
    logger.info("処理を開始します。CSVファイル: %s", args.csv_file)

    try:
        companies = read_companies(args.csv_file)
    except (CSVFormatError, FileNotFoundError) as exc:
        logger.error("CSVファイルの読み込みに失敗しました: %s", exc)
        print(f"CSVファイルの読み込みに失敗しました: {exc}", file=sys.stderr)
        return 1

    if not companies:
        print("CSVファイルに処理対象データがありません。")
        logger.warning("CSVファイルに処理対象データがありませんでした。")
        return 0

    try:
        provider = AIProviderFactory.create(config)
    except AIProviderError as exc:
        logger.error("AIプロバイダの初期化に失敗しました: %s", exc)
        print(f"AIプロバイダの初期化に失敗しました: {exc}", file=sys.stderr)
        return 1

    batch_size = config["batch"]["size"]
    template = config["prompt"]["template"]
    followup_config = config.get("followup", {})

    batches = list(batch_companies(companies, batch_size))
    total_batches = len(batches)
    print(f"対象件数: {len(companies)}件 / "
          f"バッチサイズ: {batch_size}件 / バッチ数: {total_batches}")

    for idx, batch in enumerate(batches, start=1):
        process_batch(idx, total_batches, batch, provider, template,
                      followup_config, logger)

    print("\n=== 全ての処理が完了しました。 ===")
    logger.info("全ての処理が完了しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
