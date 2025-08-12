# 現場で活用するためのAIエージェント実践入門 - Chapter 4

このディレクトリは、書籍「現場で活用するためのAIエージェント実践入門」（講談社）の第4章に関連するソースコードとリソースを含んでいます。

4章記載のコードを実行するためには、以下の手順に従ってください。

## 前提条件

このプロジェクトを実行するには、以下の準備が必要です：

- Python 3.12 以上
- Docker および Docker Compose
- VSCode
- VSCodeのMulti-root Workspaces機能を使用し、ワークスペースとして開いている（やり方は[こちら](../README.md)を参照）
- OpenAIのアカウントとAPIキー

また、Python の依存関係は `pyproject.toml` に記載されています。

## 環境構築

### 1. chapter4のワークスペースを開く
chapter4 ディレクトリに仮想環境を作成します。
VSCode の ターミナルの追加で`chapter4` を選択します。

### 2. uvのインストール

依存関係の解決には`uv`を利用します。
`uv`を使ったことがない場合、以下の方法でインストールしてください。

`pip`を使う場合：
```bash
pip install uv
```

MacまたはLinuxの場合：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Python 仮想環境の作成と依存関係のインストール

依存関係のインストール
```bash
uv sync
```

インストール後に作成した仮想環境をアクティブにします。

```bash
source .venv/bin/activate
```

### 4. 環境変数のセット
`.env` ファイルを作成し、以下の内容を追加します。

OpenAI APIキーを持っていない場合は、[OpenAIの公式サイト](https://platform.openai.com/)から取得してください。

#### OpenAI APIを使用する場合
```env
# APIプロバイダーの選択
API_PROVIDER="openai"

# OpenAI API設定
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE="https://api.openai.com/v1"
OPENAI_MODEL="gpt-4o-2024-08-06"
```

#### Azure OpenAI APIを使用する場合
```env
# APIプロバイダーの選択
API_PROVIDER="azure"

# OpenAI設定（必須項目のため設定、Azure使用時は実際には使用されない）
OPENAI_API_KEY="dummy"
OPENAI_MODEL="gpt-4o-2024-08-06"

# Azure OpenAI設定
AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
AZURE_OPENAI_API_KEY="your_azure_api_key"
AZURE_OPENAI_API_VERSION="2024-02-15-preview"
# LLM用デプロイメント名（例: gpt-4o, gpt-4.1 など）
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"
# 埋め込み用デプロイメント名（例: text-embedding-3-small など）
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="text-embedding-3-small"
```

> ※ LLM用（AZURE_OPENAI_DEPLOYMENT_NAME）と埋め込み用（AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME）はAzureポータルでそれぞれデプロイしたモデルのデプロイメント名を設定してください。

**注意**: Azure OpenAI APIを使用する場合は、Azureポータルでリソースを作成し、適切なモデルをデプロイする必要があります。

### 5. 検索インデックスの構築

makeコマンドを使用します。

```bash
#コンテナの起動
make start.engine

#インデックスの構築
make create.index
```

`create.index`実行時にElasticsearchのコンテナでエラーが発生する場合は、`docker-compose.yml`の以下の行をコメントアウトしてください。
コメントアウトした場合、Elasticsearchのデータは永続化されないため、コンテナを削除した場合に再度インデックスを構築する必要があります。

```yaml
    volumes:
      - ./.rag_data/es_data:/usr/share/elasticsearch/data
```

あるいは、以下を実行

```bash
sudo chown -R 1000:0 ./.rag_data/
sudo chown -R 1000:0 ./.rag_data/es_data/
sudo chmod -R g+rwX ./.rag_data/
sudo chmod -R g+rwX ./.rag_data/es_data/
```
