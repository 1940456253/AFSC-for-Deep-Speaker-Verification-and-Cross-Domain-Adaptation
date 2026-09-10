# AFSC + ECAPA-TDNN: トレーニングと話者照合

**Adaptive Frequency Spectral Coefficients for Deep Speaker Verification: Training and Cross-Domain Adaptation** の第一部分のスタンドアロン実装です。

このリリースでは、AFSC フロントエンドと、ECAPA-TDNN によるトレーニング、埋め込み抽出、評価パイプラインを提供します。MFCC と FBank は、同じ encoder を使うベースラインとして選択できます。Res2Net、X-Vector、ターゲットドメイン適応は、このリリースの範囲外です。本リポジトリは研究用データセットや学習済み研究チェックポイントを再配布せず、識別不能な過去のチェックポイントや論文中のすべての数値結果を再現することを主張しません。

## 主なディレクトリ

| ディレクトリ | 内容 |
|---|---|
| `Dataset` | 音声/CSV の読み込み、wav.scp + utt2spk 変換、合成デモ生成 |
| `Features` | パワースペクトル、MFCC/FBank、学習可能 AFSC、フィルタ出力 |
| `Model` | ECAPA、コサイン分類器、角度マージン損失、二段階トレーニング、YAML レシピ |
| `Evaluation` | 試行スコアリング、EER/MinDCF、ユニットテストと統合テスト |
| `Inference` | チェックポイント読み込み、発話埋め込み、2 波形比較 |

以下のコマンドはすべて **このリポジトリのルートディレクトリから** 実行します。Windows でも同じ `python -m ...` コマンドが動作します。マルチプロセスが不便な場合は `num_workers: 0` を使用してください。`speakerlab`、Kaldi バイナリ、`kaldiio` のインストールは不要です。固定フロントエンドには torchaudio の Kaldi 互換 Python 関数を使用します。

## インストール

Python 3.10–3.12 を使用してください。同梱の検証は Python 3.12、CPU で実行されています。

```bash
python -m venv .venv