"""AIプロバイダ(ChatGPT / Gemini / Claude)を切り替えるためのモジュール。

Strategyパターンにより各AIプロバイダのAPI呼び出し方法を共通インタ
フェース(AIProviderBase)に隠蔽し、Factoryパターン(AIProviderFactory)
により config.yaml の設定値(ai.provider)から適切な実装を生成する。

各プロバイダは、ブラウザで検索した場合のレスポンスとAPI経由のレスポ
ンスができるだけ一致するよう、各社が提供するWeb検索(グラウンディング)
ツールをまず有効化して呼び出し、利用できない場合は通常応答へフォール
バックする。
"""
import abc
import logging

logger = logging.getLogger("company_info_extractor")


class AIProviderError(Exception):
    """AIプロバイダ呼び出し時のエラー。"""


class AIProviderBase(abc.ABC):
    """AIプロバイダの共通インタフェース(Strategyパターンの基底クラス)。"""

    def generate(self, prompt: str) -> str:
        """プロンプトを送信し、レスポンス文字列を取得する(Template Method)。

        プロンプトおよびレスポンスはDEBUGレベルでログ出力する。

        Args:
            prompt: AIに送信するプロンプト文字列。

        Returns:
            AIからのレスポンス文字列(生テキスト)。

        Raises:
            AIProviderError: API呼び出しに失敗した場合。
        """
        logger.debug("送信プロンプト:\n%s", prompt)
        try:
            response_text = self._call_api(prompt)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("AI API呼び出しでエラーが発生しました。")
            raise AIProviderError(str(exc)) from exc
        logger.debug("受信レスポンス:\n%s", response_text)
        return response_text

    @abc.abstractmethod
    def _call_api(self, prompt: str) -> str:
        """各プロバイダ固有のAPI呼び出し処理。

        Args:
            prompt: AIに送信するプロンプト文字列。

        Returns:
            AIからのレスポンス文字列(生テキスト)。
        """
        raise NotImplementedError


class OpenAIProvider(AIProviderBase):
    """OpenAI(ChatGPT)を利用するプロバイダ実装。"""

    def __init__(self, settings: dict):
        """OpenAIProviderを初期化する。

        Args:
            settings: config.yaml の ai.openai セクション。

        Raises:
            AIProviderError: openai パッケージが未インストールの場合。
        """
        try:
            import openai
        except ImportError as exc:
            raise AIProviderError(
                "openai パッケージがインストールされていません。"
                "'pip install openai' を実行してください。") from exc

        self._client = openai.OpenAI(api_key=settings.get("api_key"))
        self._model = settings.get("model", "gpt-4o")
        self._temperature = settings.get("temperature", 0.0)

    def _call_api(self, prompt: str) -> str:
        """OpenAI APIを呼び出す。

        ブラウザ経由の検索結果とレスポンスを揃えるため、まず
        web_search ツールを有効化した Responses API を試行し、
        非対応/失敗時は通常の Chat Completions にフォールバックする。

        Args:
            prompt: AIに送信するプロンプト文字列。

        Returns:
            AIからのレスポンス文字列。
        """
        try:
            response = self._client.responses.create(
                model=self._model,
                input=prompt,
                tools=[{"type": "web_search"}],
                temperature=self._temperature,
            )
            return response.output_text
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "web_search ツール利用に失敗したため、通常応答に"
                "フォールバックします。詳細: %s", exc)
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
            )
            return completion.choices[0].message.content


class GeminiProvider(AIProviderBase):
    """Google Gemini を利用するプロバイダ実装。"""

    def __init__(self, settings: dict):
        """GeminiProviderを初期化する。

        Args:
            settings: config.yaml の ai.gemini セクション。

        Raises:
            AIProviderError: google-generativeai パッケージが
                未インストールの場合。
        """
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise AIProviderError(
                "google-generativeai パッケージがインストールされて"
                "いません。'pip install google-generativeai' を"
                "実行してください。") from exc

        genai.configure(api_key=settings.get("api_key"))
        self._genai = genai
        self._model_name = settings.get("model", "gemini-1.5-pro")

    def _call_api(self, prompt: str) -> str:
        """Gemini APIを呼び出す。

        ブラウザ検索相当の最新情報を反映するため、Google検索連携
        (grounding)ツールを有効化して呼び出しを試行し、非対応/失敗時
        は通常応答にフォールバックする。

        Args:
            prompt: AIに送信するプロンプト文字列。

        Returns:
            AIからのレスポンス文字列。
        """
        try:
            model = self._genai.GenerativeModel(
                self._model_name, tools="google_search_retrieval")
            response = model.generate_content(prompt)
            return response.text
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Google検索連携ツール利用に失敗したため、通常応答に"
                "フォールバックします。詳細: %s", exc)
            model = self._genai.GenerativeModel(self._model_name)
            response = model.generate_content(prompt)
            return response.text


class ClaudeProvider(AIProviderBase):
    """Anthropic Claude を利用するプロバイダ実装。"""

    def __init__(self, settings: dict):
        """ClaudeProviderを初期化する。

        Args:
            settings: config.yaml の ai.claude セクション。

        Raises:
            AIProviderError: anthropic パッケージが未インストールの場合。
        """
        try:
            import anthropic
        except ImportError as exc:
            raise AIProviderError(
                "anthropic パッケージがインストールされていません。"
                "'pip install anthropic' を実行してください。") from exc

        self._client = anthropic.Anthropic(api_key=settings.get("api_key"))
        self._model = settings.get("model", "claude-sonnet-4-6")
        self._max_tokens = settings.get("max_tokens", 4096)

    def _call_api(self, prompt: str) -> str:
        """Claude APIを呼び出す。

        ブラウザ検索相当の最新情報を反映するため、web_search ツール
        を有効化して呼び出しを試行し、非対応/失敗時は通常応答に
        フォールバックする。

        Args:
            prompt: AIに送信するプロンプト文字列。

        Returns:
            AIからのレスポンス文字列。
        """
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "web_search_20250305",
                        "name": "web_search"}],
            )
            return self._extract_text(response)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "web_search ツール利用に失敗したため、通常応答に"
                "フォールバックします。詳細: %s", exc)
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Claude APIレスポンスからテキスト部分のみを抽出する。

        Args:
            response: Anthropic APIのレスポンスオブジェクト。

        Returns:
            テキストブロックを連結した文字列。
        """
        texts = [
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        return "\n".join(texts)


class AIProviderFactory:
    """設定内容から適切なAIProviderインスタンスを生成するFactory。"""

    _PROVIDERS = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "claude": ClaudeProvider,
    }

    @classmethod
    def create(cls, config: dict) -> AIProviderBase:
        """設定に応じたAIProviderインスタンスを生成する。

        Args:
            config: config.yaml 全体の辞書。

        Returns:
            AIProviderBase を実装したプロバイダインスタンス。

        Raises:
            AIProviderError: 未対応のプロバイダが指定された場合。
        """
        provider_name = config["ai"]["provider"]
        provider_cls = cls._PROVIDERS.get(provider_name)
        if provider_cls is None:
            raise AIProviderError(
                f"未対応のAIプロバイダです: {provider_name} "
                f"(対応: {list(cls._PROVIDERS.keys())})")

        settings = config["ai"].get(provider_name, {})
        logger.info("AIプロバイダ '%s' を使用します。", provider_name)
        return provider_cls(settings)
