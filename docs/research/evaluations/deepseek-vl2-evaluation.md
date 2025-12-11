# DeepSeek-VL2 評価レポート

**評価日時**: 2025-12-11
**評価対象**: DeepSeek-VL2-Tiny (deepseek-ai/deepseek-vl2-tiny)
**評価目的**: Issue #31 ブロック分類改善のためのOCR/レイアウト検出ツール評価

## 1. 概要

DeepSeek-VL2は、DeepSeek社が開発したVision-Language Model (VLM)で、画像理解とテキスト生成を組み合わせた機能を提供します。OCR機能とレイアウト検出機能を持ち、学術論文PDFの翻訳に適用可能かを評価しました。

## 2. モデル情報

| 項目 | 詳細 |
|------|------|
| モデル名 | DeepSeek-VL2-Tiny |
| モデルID | deepseek-ai/deepseek-vl2-tiny |
| パラメータ数 | 約3.37B |
| VRAM使用量 | 約6.3GB (BF16) |
| ライセンス | MIT (コード), DeepSeek License (モデル) |
| リポジトリ | https://github.com/deepseek-ai/DeepSeek-VL2 |

## 3. 環境・互換性

### 3.1 要求環境

DeepSeek-VL2の公式要件：
```
torch==2.0.1
transformers==4.38.2
xformers>=0.0.21
timm>=0.9.16
accelerate
sentencepiece
attrdict
einops
```

### 3.2 互換性問題

**重大な問題**: DeepSeek-VL2は`transformers==4.38.2`を厳格に要求し、新しいバージョン（例: 4.57.3）では動作しません。

発見された問題：

1. **`LlamaFlashAttention2`の削除**
   - transformers 4.50以降で`LlamaFlashAttention2`クラスが削除された
   - 影響: モジュールのインポート失敗
   - 対処: 条件付きインポートでパッチ可能

2. **`GenerationMixin`の継承変更**
   - transformers 4.50以降で`PreTrainedModel`が`GenerationMixin`を継承しなくなった
   - 影響: `model.generate()`メソッドが使用不可
   - 対処: クラス定義の修正が必要（複雑）

3. **`generation_config`の初期化**
   - 新しいtransformersでは`generation_config`が必須
   - 影響: generate呼び出し時にAttributeError
   - 対処: モデルロード後に手動で設定（部分的に解決）

4. **Config属性の不一致**
   - `DeepseekVLV2Config`に`num_hidden_layers`属性がない
   - 影響: DynamicCacheの初期化失敗
   - 対処: 根本的なコード修正が必要

### 3.3 パッチ適用の試み

以下のパッチを適用したが、完全な互換性は達成できなかった：

```python
# modeling_deepseek.py パッチ
from transformers.models.llama.modeling_llama import LlamaAttention
try:
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
except ImportError:
    LlamaFlashAttention2 = None

class DeepseekV2PreTrainedModel(PreTrainedModel, GenerationMixin):
    # ...
```

## 4. ライセンス互換性

| ライセンス | 対象 | AGPL-3.0との互換性 |
|------------|------|-------------------|
| MIT | コード | ✅ 互換 |
| DeepSeek License | モデル重み | ⚠️ 要確認 |

DeepSeek Licenseはモデル重みに適用され、商用利用に制限がある可能性があります。

## 5. 評価結果

### 5.1 インストール・セットアップ

| 項目 | 結果 |
|------|------|
| インストール | ⚠️ 複雑（バージョン固定必要） |
| GPUメモリ | ✅ 12GB GPUで動作可能 |
| 依存関係 | ❌ transformers 4.38.2固定 |
| モデルロード | ✅ 成功（パッチ適用後） |
| 推論実行 | ❌ 失敗（互換性問題） |

### 5.2 機能評価（推定）

公式ドキュメントとデモに基づく推定評価：

| 機能 | 期待される能力 |
|------|---------------|
| OCRテキスト抽出 | ✅ 高精度 |
| レイアウト検出 | ✅ 可能（プロンプトベース） |
| 座標情報取得 | ⚠️ グラウンディング機能で可能 |
| 処理速度 | ⚠️ VLMのため低速（推定1-2ページ/分） |

### 5.3 Issue #31への適用性

| 評価項目 | スコア | コメント |
|----------|--------|----------|
| 見出し検出 | N/A | 評価不可（互換性問題） |
| 本文/キャプション分類 | N/A | 評価不可 |
| 座標情報取得 | N/A | 評価不可 |
| 処理速度 | ⚠️ | VLMのため低速と推定 |
| 統合の容易さ | ❌ | 依存関係の制約が大きい |

## 6. 他ツールとの比較

| ツール | 速度 | 精度 | 座標情報 | 統合容易性 | 推奨度 |
|--------|------|------|----------|------------|--------|
| PyMuPDF4LLM | ~1.3p/s | 良好 | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| DocLayout-YOLO | ~7p/s | 良好 | ✅ | ✅ | ⭐⭐⭐⭐ |
| DeepSeek-VL2 | ~0.5p/min(推定) | 高(推定) | ⚠️ | ❌ | ⭐⭐ |

## 7. 結論と推奨事項

### 7.1 結論

DeepSeek-VL2は高度なVision-Language能力を持つ有望なモデルですが、以下の理由によりIssue #31での採用は**推奨しません**：

1. **依存関係の制約**: transformers 4.38.2固定は、プロジェクトの他の依存関係と競合する可能性が高い
2. **処理速度**: VLMベースのため、PyMuPDF4LLMやDocLayout-YOLOと比較して大幅に低速
3. **統合の複雑さ**: パッチ適用やバージョン固定が必要で、保守コストが高い

### 7.2 推奨事項

Issue #31のブロック分類改善には以下を推奨：

1. **第一選択**: PyMuPDF4LLM
   - `to_json()`による座標情報取得
   - `boxclass`によるヘッダー/フッター検出
   - プロジェクトの既存PyMuPDF使用との親和性

2. **補助ツール**: DocLayout-YOLO
   - 高速なレイアウト検出（7ページ/秒）
   - 10種類のブロック分類
   - PyMuPDF4LLMと組み合わせて使用可能

### 7.3 今後の検討

DeepSeek-VL2を将来的に再評価する場合：
- transformers 4.38.2を使用する専用環境を構築
- または、DeepSeek社が新しいtransformersバージョンに対応するのを待つ
- 代替として、GOT-OCR2.0やPaddleOCRなど他のOCRツールを評価

## 8. 評価環境

```
OS: Linux 6.14.0-36-generic
GPU: NVIDIA GeForce RTX 4070 Ti (12GB)
Python: 3.12
torch: 2.9.1
transformers: 4.57.3 (互換性問題あり)
```

## 9. 参考資料

- [DeepSeek-VL2 GitHub](https://github.com/deepseek-ai/DeepSeek-VL2)
- [DeepSeek-VL2 Paper](https://arxiv.org/abs/2412.10302)
- [HuggingFace Model](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny)
