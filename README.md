# 会社情報AI調査ツール

会社名・住所・電話番号のリストから、AI(ChatGPT / Gemini / Claude を切替可能)
を用いて公式ホームページURL・公式メールアドレス・Instagram公式アカウント・
Facebook公式アカウント・LINE公式アカウントを調査するツールです。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `main.py` | エントリポイント。CSV読込→バッチ分割→AI問い合わせ→結果表示を統括 |
| `config.yaml` | AIプロバイダ設定、プロンプト雛形、バッチ件数、ログ設定 |
| `ai_providers.py` | ChatGPT/Gemini/Claude 切替(Strategy+Factoryパターン) |
| `csv_handler.py` | 入力CSVの読込・バッチ分割 |
| `prompt_builder.py` | プロンプト生成 |
| `response_parser.py` | AIレスポンスからJSONを抽出 |
| `config_loader.py` | config.yaml の読込・検証 |
| `logger_setup.py` | ログ設定(標準出力+ファイル出力、DEBUGレベル) |
| `sample_companies.csv` | 入力CSVのサンプル |
| `test_main.py` | 単体テスト・結合テスト(pytest) |

## セットアップ

```bash
pip install -r requirements.txt
```

利用するAIプロバイダのSDKのみで構いません(例: OpenAIのみ使う場合は
`pip install PyYAML openai`)。

`config.yaml` に、利用するプロバイダの `api_key` 等を設定してください。

```yaml
ai:
  provider: openai   # openai / gemini / claude から選択
  openai:
    api_key: "sk-xxxx..."
    model: "gpt-4o"
```

## 入力CSVフォーマット

`company_name`, `address`, `phone` の3列を持つCSVファイルを用意してください
(`sample_companies.csv` 参照)。

```csv
company_name,address,phone
株式会社サンプル商事,東京都千代田区丸の内1-1-1,03-1234-5678
```

## 実行方法

```bash
python main.py sample_companies.csv --config config.yaml
```

- `config.yaml` の `batch.size` で指定した件数ずつ、AIへまとめて問い合わせます。
- バッチごとに進捗(`バッチ N/M を処理中...`)と結果を都度標準出力します。
- 取得できなかった項目は「取得できませんでした」と表示されます。
- 送信プロンプト・AIレスポンスはDEBUGレベルで
  `logging.log_file`(既定: `logs/app.log`)に記録されます。

## テスト実行

```bash
pip install pytest
pytest -v test_main.py
```

CSV読込、プロンプト生成、JSON抽出(コードブロック付き/なし、埋め込み文中、
不正データ等)、設定ファイル検証、AI呼び出し失敗・レスポンス解析失敗時の
異常系ハンドリングまで、モックAIプロバイダを用いて23件のテストで検証済みです。

## 設計上のポイント

- **Strategyパターン**: `AIProviderBase` を共通インタフェースとし、
  `OpenAIProvider` / `GeminiProvider` / `ClaudeProvider` が実装を差し替え。
- **Factoryパターン**: `AIProviderFactory.create()` が `config.yaml` の
  `ai.provider` 値に応じて適切な実装を生成。
- **ブラウザ結果との整合性**: 各プロバイダで、可能な場合はWeb検索/グラウン
  ディングツール(OpenAIのweb_search、Geminiのgoogle_search_retrieval、
  Claudeのweb_search_20250305)に加え、Claudeでは公式サイトを実際に取得
  (フェッチ)できる `web_fetch`(beta)も有効化し、非対応/失敗時は段階的に
  フォールバックします。これにより、検索スニペットだけでなく実際のサイト
  内容を確認でき、ブラウザ版に近い結果が得られやすくなります。
- **未確認項目の再調査(followup)**: アイコンボタン等のSNSリンクは、
  遷移先URL(href)が本文テキストとして見えないことがあり、AIが慎重すぎて
  `null`と回答してしまうケースがあります。初回調査で
  Instagram/Facebook/LINE/メールアドレスが未確認だった会社について、
  「リンク先URLを確認する」「他ページも見る」「グループ名でも検索する」
  「表記ゆれを変えて再検索する」といった具体的な指示を含む追加プロンプト
  で1社ずつ再調査し、見つかった項目のみ結果にマージします
  (`config.yaml` の `followup.enabled` で有効/無効を切替可能)。
- **JSON抽出の頑健性**: AIが説明文やコードブロックを付与して返しても、
  複数パターンで抽出を試行しJSONを取り出します。
- **エラー時も処理継続**: 1バッチのAI呼び出し失敗・JSON解析失敗があっても、
  そのバッチの対象企業は「取得できませんでした」として表示し、後続バッチ
  の処理は継続します。
- **デバッグ用ログの充実**: Claudeプロバイダでは、実際に投げられた検索/
  取得クエリ、`stop_reason`(max_tokensによる打ち切り検知を含む)を
  DEBUGログに記録し、ブラウザ版との差異調査を容易にしています。
