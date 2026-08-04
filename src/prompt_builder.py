"""AIに送信するプロンプトを生成するモジュール。"""
from typing import Dict, List


def build_prompt(template: str, companies: List[Dict[str, str]]) -> str:
    """config.yaml のプロンプト雛形を用いて実際のプロンプトを生成する。

    Args:
        template: プロンプトの雛形文字列。'{company_list}' プレース
            ホルダを含む必要がある。JSON例示部分の中括弧は雛形側で
            '{{' '}}' のように二重化しておくこと。
        companies: 会社名、住所、電話番号を保持する辞書のリスト。

    Returns:
        AIに送信するプロンプト文字列。

    Raises:
        KeyError: テンプレートに '{company_list}' 以外の未知の
            プレースホルダが含まれる場合。
    """
    lines = []
    for idx, company in enumerate(companies, start=1):
        lines.append(
            f"{idx}. 会社名: {company['company_name']} / "
            f"住所: {company['address']} / 電話番号: {company['phone']}"
        )
    company_list_text = "\n".join(lines)

    return template.format(company_list=company_list_text)


def build_followup_prompt(
        template: str, company: Dict[str, str],
        missing_field_labels: List[str]) -> str:
    """未確認項目がある1社について、再調査用プロンプトを生成する。

    Args:
        template: config.yaml の followup.template。'{company_name}'
            '{address}' '{phone}' '{missing_fields}' プレースホルダを
            含む必要がある。JSON例示部分の中括弧は雛形側で '{{' '}}'
            のように二重化しておくこと。
        company: 対象の会社情報(company_name/address/phone)。
        missing_field_labels: 未確認項目の日本語ラベルのリスト
            (例: ["Instagram公式アカウント", "公式メールアドレス"])。

    Returns:
        AIに送信する再調査用プロンプト文字列。
    """
    return template.format(
        company_name=company["company_name"],
        address=company["address"],
        phone=company["phone"],
        missing_fields="、".join(missing_field_labels),
    )
