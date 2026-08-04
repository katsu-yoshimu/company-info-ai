"""単体テスト・結合テスト。

pytest 実行:
    pytest -v test_main.py
"""
import json
import os
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_providers import AIProviderBase, AIProviderError  # noqa: E402
from config_loader import ConfigError, load_config  # noqa: E402
from csv_handler import (CSVFormatError, batch_companies,  # noqa: E402
                          read_companies)
from prompt_builder import build_prompt  # noqa: E402
from response_parser import ResponseParseError, extract_json  # noqa: E402
import main as main_module  # noqa: E402


SAMPLE_TEMPLATE = textwrap.dedent("""\
    以下を調査してください。
    {company_list}
    JSON配列のみで返してください。
    [
      {{
        "company_name": "会社名",
        "website_url": "URLまたはnull"
      }}
    ]
    """)


class MockAIProvider(AIProviderBase):
    """テスト用のモックAIプロバイダ。"""

    def __init__(self, response_text=None, raise_error=False):
        self._response_text = response_text
        self._raise_error = raise_error
        self.received_prompts = []

    def _call_api(self, prompt: str) -> str:
        self.received_prompts.append(prompt)
        if self._raise_error:
            raise RuntimeError("mock api error")
        return self._response_text


class TestCSVHandler(unittest.TestCase):
    """csv_handler モジュールのテスト。"""

    def test_read_companies_success(self):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False,
                encoding="utf-8") as f:
            f.write("company_name,address,phone\n")
            f.write("株式会社A,東京都千代田区1-1,03-1111-1111\n")
            f.write("株式会社B,大阪府大阪市2-2,06-2222-2222\n")
            path = f.name

        try:
            companies = read_companies(path)
            self.assertEqual(len(companies), 2)
            self.assertEqual(companies[0]["company_name"], "株式会社A")
            self.assertEqual(companies[1]["phone"], "06-2222-2222")
        finally:
            os.remove(path)

    def test_read_companies_missing_column(self):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False,
                encoding="utf-8") as f:
            f.write("company_name,address\n")
            f.write("株式会社A,東京都\n")
            path = f.name

        try:
            with self.assertRaises(CSVFormatError):
                read_companies(path)
        finally:
            os.remove(path)

    def test_read_companies_empty_name(self):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False,
                encoding="utf-8") as f:
            f.write("company_name,address,phone\n")
            f.write(",東京都,03-1111-1111\n")
            path = f.name

        try:
            with self.assertRaises(CSVFormatError):
                read_companies(path)
        finally:
            os.remove(path)

    def test_read_companies_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_companies("/no/such/file.csv")

    def test_batch_companies(self):
        companies = [{"company_name": f"社{i}", "address": "", "phone": ""}
                     for i in range(7)]
        batches = list(batch_companies(companies, 3))
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[1]), 3)
        self.assertEqual(len(batches[2]), 1)

    def test_batch_companies_invalid_size(self):
        with self.assertRaises(ValueError):
            list(batch_companies([{"company_name": "A"}], 0))


class TestPromptBuilder(unittest.TestCase):
    """prompt_builder モジュールのテスト。"""

    def test_build_prompt_contains_company_info(self):
        companies = [
            {"company_name": "株式会社A", "address": "東京都",
             "phone": "03-1111-1111"},
            {"company_name": "株式会社B", "address": "大阪府",
             "phone": "06-2222-2222"},
        ]
        prompt = build_prompt(SAMPLE_TEMPLATE, companies)
        self.assertIn("株式会社A", prompt)
        self.assertIn("東京都", prompt)
        self.assertIn("株式会社B", prompt)
        self.assertIn('"company_name": "会社名"', prompt)


class TestResponseParser(unittest.TestCase):
    """response_parser モジュールのテスト。"""

    def test_extract_json_plain(self):
        text = json.dumps([{"company_name": "A", "website_url": "http://a"}])
        result = extract_json(text)
        self.assertEqual(result[0]["company_name"], "A")

    def test_extract_json_with_code_fence(self):
        payload = json.dumps([{"company_name": "A"}])
        text = f"はい、結果です。\n```json\n{payload}\n```\n以上です。"
        result = extract_json(text)
        self.assertEqual(result[0]["company_name"], "A")

    def test_extract_json_with_plain_fence(self):
        payload = json.dumps([{"company_name": "B"}])
        text = f"```\n{payload}\n```"
        result = extract_json(text)
        self.assertEqual(result[0]["company_name"], "B")

    def test_extract_json_embedded_in_text(self):
        payload = json.dumps([{"company_name": "C"}])
        text = f"調査結果は以下の通りです: {payload} ご確認ください。"
        result = extract_json(text)
        self.assertEqual(result[0]["company_name"], "C")

    def test_extract_json_single_object(self):
        text = json.dumps({"company_name": "D"})
        result = extract_json(text)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["company_name"], "D")

    def test_extract_json_invalid(self):
        with self.assertRaises(ResponseParseError):
            extract_json("これはJSONではありません。")

    def test_extract_json_empty(self):
        with self.assertRaises(ResponseParseError):
            extract_json("")


class TestConfigLoader(unittest.TestCase):
    """config_loader モジュールのテスト。"""

    def _write_config(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False,
                encoding="utf-8") as f:
            f.write(content)
            return f.name

    def test_load_valid_config(self):
        path = self._write_config(textwrap.dedent("""\
            ai:
              provider: openai
              openai:
                api_key: dummy
            batch:
              size: 5
            prompt:
              template: "調査対象: {company_list}"
            logging:
              log_file: logs/test.log
            """))
        try:
            config = load_config(path)
            self.assertEqual(config["ai"]["provider"], "openai")
            self.assertEqual(config["batch"]["size"], 5)
        finally:
            os.remove(path)

    def test_load_config_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config("/no/such/config.yaml")

    def test_load_config_missing_required_key(self):
        path = self._write_config("ai:\n  provider: openai\n")
        try:
            with self.assertRaises(ConfigError):
                load_config(path)
        finally:
            os.remove(path)

    def test_load_config_missing_placeholder(self):
        path = self._write_config(textwrap.dedent("""\
            ai:
              provider: openai
              openai:
                api_key: dummy
            batch:
              size: 5
            prompt:
              template: "プレースホルダなし"
            logging:
              log_file: logs/test.log
            """))
        try:
            with self.assertRaises(ConfigError):
                load_config(path)
        finally:
            os.remove(path)


class TestMainIntegration(unittest.TestCase):
    """main モジュールの結合テスト(モックAIプロバイダを使用)。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        self.csv_path = os.path.join(self.tmp_dir, "companies.csv")
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("company_name,address,phone\n")
            f.write("株式会社A,東京都千代田区1-1,03-1111-1111\n")
            f.write("株式会社B,大阪府大阪市2-2,06-2222-2222\n")
            f.write("株式会社C,広島県広島市3-3,082-333-3333\n")

        self.config_path = os.path.join(self.tmp_dir, "config.yaml")
        log_path = os.path.join(self.tmp_dir, "app.log")
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(f"""\
                ai:
                  provider: openai
                  openai:
                    api_key: dummy
                batch:
                  size: 2
                prompt:
                  template: |
                    調査対象: {{company_list}}
                    JSONで返却してください。
                logging:
                  log_file: "{log_path}"
                  console_level: "DEBUG"
                  file_level: "DEBUG"
                """))

    def test_main_success_with_missing_fields(self):
        """一部項目が取得できないケースを含む正常系。"""
        mock_response = json.dumps([
            {
                "company_name": "株式会社A",
                "website_url": "https://a-corp.example.com",
                "email": "info@a-corp.example.com",
                "instagram": "https://instagram.com/a_corp",
                "facebook": None,
                "line": None,
            },
            {
                "company_name": "株式会社B",
                "website_url": None,
                "email": None,
                "instagram": None,
                "facebook": None,
                "line": None,
            },
        ])
        mock_response_2 = json.dumps([
            {
                "company_name": "株式会社C",
                "website_url": "https://c-corp.example.com",
                "email": "contact@c-corp.example.com",
                "instagram": None,
                "facebook": "https://facebook.com/c_corp",
                "line": "https://line.me/R/ti/p/@c_corp",
            },
        ])

        call_results = [mock_response, mock_response_2]

        def fake_create(config):
            provider = MockAIProvider(response_text=None)
            call_iter = iter(call_results)

            def _call_api(prompt):
                provider.received_prompts.append(prompt)
                return next(call_iter)

            provider._call_api = _call_api
            return provider

        with patch("main.AIProviderFactory.create", side_effect=fake_create):
            with patch("sys.stdout") as mock_stdout:
                ret_code = main_module.main(
                    [self.csv_path, "--config", self.config_path])

        self.assertEqual(ret_code, 0)
        output = "".join(
            call.args[0] for call in mock_stdout.write.call_args_list
            if call.args and isinstance(call.args[0], str))
        self.assertIn("株式会社A", output)
        self.assertIn("https://a-corp.example.com", output)
        self.assertIn("取得できませんでした", output)
        self.assertIn("株式会社C", output)
        self.assertIn("https://line.me/R/ti/p/@c_corp", output)

    def test_main_ai_error_is_handled_gracefully(self):
        """AI呼び出しが失敗しても処理全体が異常終了しないことを確認。"""

        def fake_create(config):
            return MockAIProvider(raise_error=True)

        with patch("main.AIProviderFactory.create", side_effect=fake_create):
            with patch("sys.stdout") as mock_stdout:
                ret_code = main_module.main(
                    [self.csv_path, "--config", self.config_path])

        self.assertEqual(ret_code, 0)
        output = "".join(
            call.args[0] for call in mock_stdout.write.call_args_list
            if call.args and isinstance(call.args[0], str))
        self.assertIn("スキップします", output)
        self.assertIn("取得できませんでした", output)

    def test_main_invalid_json_response_is_handled_gracefully(self):
        """AIが不正なレスポンスを返しても処理全体が継続することを確認。"""

        def fake_create(config):
            return MockAIProvider(response_text="これはJSONではありません")

        with patch("main.AIProviderFactory.create", side_effect=fake_create):
            with patch("sys.stdout") as mock_stdout:
                ret_code = main_module.main(
                    [self.csv_path, "--config", self.config_path])

        self.assertEqual(ret_code, 0)
        output = "".join(
            call.args[0] for call in mock_stdout.write.call_args_list
            if call.args and isinstance(call.args[0], str))
        self.assertIn("解析に失敗した", output)

    def test_main_followup_recovers_missing_sns_field(self):
        """初回nullの項目をフォローアップで拾い直せることを確認する。"""
        followup_config_path = os.path.join(
            self.tmp_dir, "config_followup.yaml")
        log_path = os.path.join(self.tmp_dir, "app_followup.log")
        with open(followup_config_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(f"""\
                ai:
                  provider: openai
                  openai:
                    api_key: dummy
                batch:
                  size: 5
                prompt:
                  template: |
                    調査対象: {{company_list}}
                    JSONで返却してください。
                followup:
                  enabled: true
                  template: |
                    再調査対象: {{company_name}} / {{address}} / {{phone}}
                    未確認項目: {{missing_fields}}
                    JSONオブジェクトのみで返却してください。
                logging:
                  log_file: "{log_path}"
                  console_level: "DEBUG"
                  file_level: "DEBUG"
                """))

        csv_path = os.path.join(self.tmp_dir, "one_company.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("company_name,address,phone\n")
            f.write(
                "赤坂デンタルクリニック,東京都港区赤坂3-13-13,"
                "03-3585-1548\n")

        first_pass_response = json.dumps([{
            "company_name": "赤坂デンタルクリニック",
            "website_url": "https://akasaka-dental.com/",
            "email": None,
            "instagram": None,
            "facebook": None,
            "line": None,
        }])
        followup_response = json.dumps({
            "company_name": "赤坂デンタルクリニック",
            "email": None,
            "instagram": "https://www.instagram.com/akasaka_dental/",
            "facebook": None,
            "line": None,
        })

        call_log = []

        def fake_create(config):
            provider = MockAIProvider()
            responses = iter([first_pass_response, followup_response])

            def _call_api(prompt):
                call_log.append(prompt)
                return next(responses)

            provider._call_api = _call_api
            return provider

        with patch("main.AIProviderFactory.create", side_effect=fake_create):
            with patch("sys.stdout") as mock_stdout:
                ret_code = main_module.main(
                    [csv_path, "--config", followup_config_path])

        self.assertEqual(ret_code, 0)
        self.assertEqual(len(call_log), 2)
        self.assertIn("未確認項目", call_log[1])
        self.assertIn("Instagram公式アカウント", call_log[1])

        output = "".join(
            call.args[0] for call in mock_stdout.write.call_args_list
            if call.args and isinstance(call.args[0], str))
        self.assertIn(
            "https://www.instagram.com/akasaka_dental/", output)
        # Facebook等、再調査でも見つからなかった項目はnullのまま
        self.assertIn("Facebook公式アカウント", output)

    def test_main_csv_not_found(self):
        with patch("sys.stderr"):
            ret_code = main_module.main(
                ["/no/such/companies.csv", "--config", self.config_path])
        self.assertEqual(ret_code, 1)

    def test_main_config_not_found(self):
        with patch("sys.stderr"):
            ret_code = main_module.main(
                [self.csv_path, "--config", "/no/such/config.yaml"])
        self.assertEqual(ret_code, 1)


if __name__ == "__main__":
    unittest.main()
