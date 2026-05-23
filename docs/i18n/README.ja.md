<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png">
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/logos/vllm-omni-logo.png" width=55%>
  </picture>
</p>
<h3 align="center">
誰もが使える、簡単・高速・低コストなオムニモーダル推論サービング
</h3>

<p align="center">
| <a href="https://vllm-omni.readthedocs.io/en/latest/"><b>ドキュメント</b></a> | <a href="https://discuss.vllm.ai"><b>ユーザーフォーラム</b></a> | <a href="https://slack.vllm.ai"><b>開発者 Slack</b></a> | <a href="../assets/WeChat.jpg"><b>WeChat</b></a> | <a href="https://arxiv.org/abs/2602.02204"><b>論文</b></a> | <a href="https://docs.google.com/presentation/d/1XJWgv79lORl8rbaVvp2d5Sqs6ZEBgAgj/edit?slide=id.p1#slide=id.p1"><b>スライド</b></a> |
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.fr.md">Français</a> ·
  <b>日本語</b>
</p>

---

*最新情報* 🔥
- [2026/03] [0.18.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.18.0) をリリースしました — 大規模なエントリポイントのリファクタリングおよびスケジューラ／ランタイムのクリーンアップによりコアランタイムを強化し、統合された量子化と拡散モデルの実行を拡張し、マルチモーダルモデルのカバレッジを広げ、音声・オムニ・画像・動画・RL・マルチプラットフォーム展開全般にわたって本番運用適性を向上させました。
- [2026/03] vLLM 香港ミートアップで行った初の公開[プロジェクトディープダイブ](https://youtu.be/sgwNfsNnR9I)をぜひご覧ください！
- [2026/03] **[vllm-omni-skills](https://github.com/hsliuustc0106/vllm-omni-skills)** は、開発者が vLLM-Omni をより効果的に活用できるよう支援する、コミュニティ主導の AI アシスタントスキル集です。**Cursor IDE**、**Claude**、**Codex** など、主要なエージェント型コーディングアシスタントと組み合わせて利用できます。
- [2026/02] [0.16.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.16.0) をリリースしました — **アップストリームの vLLM v0.16.0** にリベースする大規模なアラインメント＋機能拡張リリースで、**Qwen3-Omni / Qwen3-TTS**、**Bagel**、**MiMo-Audio**、**GLM-Image**、および **拡散 (DiT) 画像／動画スタック** において、性能・分散実行・本番運用適性を大きく拡張しています。あわせて、プラットフォームカバレッジ (CUDA / ROCm / NPU / XPU)、CI 品質、ドキュメントも改善されています。
- [2026/02] [0.14.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.14.0) をリリースしました — vLLM-Omni 初の **安定版リリース** であり、Omni の拡散／画像・動画生成および音声／TTS スタックを拡張し、分散実行とメモリ効率を改善、プラットフォーム／バックエンドのカバレッジ (GPU/ROCm/NPU/XPU) を拡大しました。さらに、サービング API、プロファイリング・ベンチマーク、全体的な安定性にも意義ある改善があります。アーキテクチャ設計と性能結果については、最新の[論文](https://arxiv.org/abs/2602.02204) をご参照ください。
- [2026/01] [0.12.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.12.0rc1) をリリースしました — 拡散スタックの成熟、OpenAI 互換サービングの強化、オムニモデルのカバレッジ拡大、各プラットフォーム (GPU/NPU/ROCm) での安定性向上に焦点をあてた、主要な RC マイルストーンです。
- [2025/11] vLLM コミュニティは、オムニモーダルモデルのサービングをサポートするため、 [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) を正式に公開しました。

---

## 概要

[vLLM](https://github.com/vllm-project/vllm) は本来、テキストベースの自己回帰生成タスク向けの大規模言語モデルをサポートする目的で設計されました。vLLM-Omni はその対象を **オムニモーダルモデル** の推論・サービングへと拡張するフレームワークです:

- **オムニモーダル**: テキスト、画像、動画、音声データの処理に対応
- **非自己回帰アーキテクチャ**: vLLM の AR サポートを Diffusion Transformer (DiT) などの並列生成モデルへと拡張
- **多様な出力**: 従来のテキスト生成からマルチモーダル出力まで

<p align="center">
  <picture>
    <img alt="vllm-omni" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/omni-modality-model-architecture.png" width=55%>
  </picture>
</p>

vLLM-Omni は次の点で高速です:

- vLLM の効率的な KV キャッシュ管理を活用した、最先端の AR サポート
- パイプライン化されたステージ実行のオーバーラップによる高スループット
- OmniConnector に基づく完全分離アーキテクチャと、ステージ間の動的なリソース割り当て

vLLM-Omni は次の点で柔軟かつ使いやすいです:

- 複雑なモデルワークフローを扱う、ヘテロジニアスなパイプライン抽象
- Hugging Face の人気モデルとのシームレスな統合
- 分散推論向けの、テンソル／パイプライン／データ／エキスパート並列のサポート
- ストリーミング出力
- OpenAI 互換 API サーバー

vLLM-Omni は、HuggingFace 上の主要なオープンソースモデルの大半をシームレスにサポートしています。例:

- オムニモーダルモデル (例: Qwen-Omni)
- マルチモーダル生成モデル (例: Qwen-Image)

## はじめに

詳しくは[ドキュメント](https://vllm-omni.readthedocs.io/en/latest/) をご覧ください。

- [インストール](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)
- [クイックスタート](https://vllm-omni.readthedocs.io/en/latest/getting_started/quickstart/)
- [サポート対象モデル一覧](https://vllm-omni.readthedocs.io/en/latest/models/supported_models/)

## コントリビュート

あらゆる形のコントリビュートやコラボレーションを歓迎します。
参加方法については [Contributing to vLLM-Omni](https://vllm-omni.readthedocs.io/en/latest/contributing/) をご参照ください。

## 引用

vLLM-Omni を研究に利用される場合は、当プロジェクトの[論文](https://arxiv.org/abs/2602.02204) を引用してください:

```bibtex
@article{yin2026vllmomni,
  title={vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models},
  author={Peiqi Yin, Jiangyun Zhu, Han Gao, Chenguang Zheng, Yongxiang Huang, Taichang Zhou, Ruirui Yang, Weizhi Liu, Weiqing Chen, Canlin Guo, Didan Deng, Zifeng Mo, Cong Wang, James Cheng, Roger Wang, Hongsheng Liu},
  journal={arXiv preprint arXiv:2602.02204},
  year={2026}
}
```

## コミュニティに参加する
質問・フィードバック・他ユーザーとのディスカッションは、[slack.vllm.ai](https://slack.vllm.ai) の `#sig-omni` Slack チャンネルや、vLLM ユーザーフォーラム [discuss.vllm.ai](https://discuss.vllm.ai) でお気軽にどうぞ。

## スター履歴

[![Star History Chart](https://api.star-history.com/svg?repos=vllm-project/vllm-omni&type=date&legend=top-left)](https://www.star-history.com/#vllm-project/vllm-omni&type=date&legend=top-left)

## ライセンス

Apache License 2.0。詳細は [LICENSE](../../LICENSE) ファイルをご覧ください。

---

> **注記**: 本ファイルは英語版 [`README.md`](../../README.md) の参考訳です。英語版と内容に差異がある場合は、英語版を正としてご参照ください。
