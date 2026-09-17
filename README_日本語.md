# 深層話者照合とクロスドメイン適応のためのAFSC音声特徴

[English](README.md) | [中文](README_中文.md) | **日本語**

本プロジェクトは、AFSC 特徴量、ECAPA 学習、埋め込み抽出、話者照合、およびターゲットドメイン適応のデモを扱います。MFCC と FBank の比較も維持しています。公式の Res2Net および X-Vector コードも含まれています。コードは再学習可能で、論文で提供された CN-Celeb2 事前学習モデルの読み込みにも対応しています。クロスドメイン適応部分は現在、residual adapter、target-domain embedding 標準化、固定スコア融合の流れを検証する軽量デモを提供しています。

本プロジェクトは二つの利用方法をサポートします：

- **自分で学習**：`Speech_example/` または自分のデータセットから始め、`Model/train.py` で AFSC + ECAPA を学習します。
- **事前学習モデルを読み込む**：論文提供の `embedding_model.ckpt` + `classifier.ckpt` を `tools/convert_legacy.py` で本プロジェクトのチェックポイント形式に変換し、そのまま抽出・スコアリング・比較を行います。

両方の方法で `Inference/`、`Evaluation/`、`Features/` のスクリプトを共有します。クロスドメイン適応部分はさらに `Adaptation/` ディレクトリを使用します。

## 主要ディレクトリ

| フォルダ | 主要ファイル | 機能 |
|---|---|---|
| Dataset | data.py、prepare.py、demo.py | 音声読み込み、学習マニフェスト作成、話者フォルダからデモデータ生成 |
| Features | spectrum.py、afsc.py、frontend.py、export.py | パワースペクトルと三種の音響特徴量、学習済みフィルタの出力 |
| Model | ecapa_tdnn.py、system.py、train.py | ECAPA、分類損失、二段階学習と再開 |
| Evaluation | metrics.py、score.py、test_pipeline.py、smoke_test.py | スコアリング、EER/MinDCF、検証コード |
| Inference | extract.py、verify.py | 埋め込み抽出、二音声の比較 |
| Adaptation | adapter.py、losses.py、chunking.py、standardize.py、trials.py、sampler.py、dataset.py、embed.py、train.py、demo.py | ターゲットドメイン適応：residual adapter、embedding 標準化、固定融合、AFSC gap 更新 |

その他：

| フォルダ | 内容 |
|---|---|
| `Speech_example/` | サンプル音声。各サブフォルダが一人の話者で、各フォルダに 15 発話 |
| `PreTrained/` | 論文提供の事前学習チェックポイント（`embedding_model.ckpt` + `classifier.ckpt`）と変換後の `cn_celeb2_afsc_ecapa.pt` |
| `tools/` | 補助スクリプト。`convert_legacy.py` と `inspect_ckpt.py` を含む |

学習コードは Model 内にあるため、第六の Training ディレクトリは追加していません。すべてのコマンドはプロジェクトルートから実行します。

## インストール

Python 3.10–3.12 を使用します。まず対応する PyTorch と torchaudio 2.5.1 をインストールし、その後 requirements.txt をインストールします。CPU 版の例：

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

GPU ユーザーは対応する CUDA 版をインストールしてください。`requirements.txt` は変更不要です。

## まずデモを動かす

`Speech_example/` からマニフェストを生成します：

```bash
python -m Dataset.demo
```

期待される出力：

```
Train:  30 utterances from 3 speakers
Eval:   15 utterances from 3 speakers
Trials: 105 (30 target, 75 nontarget)
Manifests written to .../demo_data
```

`demo_data/` に生成されるファイル：

| ファイル | 内容 |
|---|---|
| `train.csv` | 学習マニフェスト。各行 `ID,path,spk` |
| `eval.csv` | 評価マニフェスト。各行 `ID,path,spk` |
| `trials.txt` | 評価 trial。各行 `enrol_id test_id label` |

デフォルトでは各話者 5 発話を評価用に残し、残りを学習に使用します。変更するには：

```bash
python -m Dataset.demo --eval-per-speaker 3
```

## 自分で学習する完全な流れ

PyCharm で以下のファイルを順に右クリック Run するか、コマンドラインから `python -m ...` で順に実行します。

### 1. 学習

```bash
python -m Model.train --config Model/demo.yaml --train-csv demo_data/train.csv --output runs/demo --device cpu
```

`Model/demo.yaml` は CPU 向けの小型モデル、`Model/config.yaml` は論文設定（2+8 epoch、`channels=[1024,1024,1024,1024,3072]`、192 次元埋め込み）で、正式な実験に使用します。

学習出力は `runs/demo/`：

| ファイル | 内容 |
|---|---|
| `config.yaml` | 今回の学習で使用した完全な設定 |
| `speakers.json` | 話者 → インデックスのマッピング |
| `train.jsonl` | 各 epoch の loss / accuracy / lr / margin |
| `epoch_XXX.pt` | 各 epoch の完全なチェックポイント |
| `last.pt` | 最後の epoch のコピー |
| `frequency_points_XXX.json` | 各 epoch の AFSC 周波数点 |

### 2. 評価セットの埋め込み抽出

```bash
python -m Inference.extract --checkpoint runs/demo/last.pt --csv demo_data/eval.csv --output runs/demo/embeddings.npz --device cpu
```

出力：

- `runs/demo/embeddings.npz`：`ids`（文字列配列）と `embeddings`（N × D）を含む
- `runs/demo/embeddings.json`：checkpoint、csv、feature、utterances、chunk_seconds を記録したメタデータ

### 3. スコアリング

```bash
python -m Evaluation.score --enrol runs/demo/embeddings.npz --trials demo_data/trials.txt --output runs/demo/scores
```

出力：

- `runs/demo/scores/scores.csv`：各 trial の enrol_id、test_id、label、score
- `runs/demo/scores/metrics.json`：EER、MinDCF、trial 数

EER の単位はパーセント、MinDCF は正規化値を使用し、デフォルトのターゲット事前確率は `0.01` です。

### 4. フィルタのエクスポート（任意）

```bash
python -m Features.export --checkpoint runs/demo/last.pt --output runs/demo/filters --plot
```

出力：

- `runs/demo/filters/afsc_filters.npz`：`bin_points` (82,)、`hz` (82,)、`filters` (80, 257) を含む
- `runs/demo/filters/afsc_filters.png`（`--plot` 指定時）

### 5. 二音声の比較（任意）

```bash
python -m Inference.verify --checkpoint runs/demo/last.pt --enrol Speech_example/speech_demo1/speech-01-001.flac --test Speech_example/speech_demo1/speech-01-002.flac
```

`--enrol` / `--test` を省略した場合、`Speech_example/speech_demo1/` の最初の二音声を使用します。`--threshold` は開発セットで校正する必要があり、汎用のデフォルト値はありません。

## クロスドメイン適応デモ

`Adaptation/` ディレクトリは、論文 3.5 節のターゲットドメイン適応手順を 3 話者の `demo_data/` 上で実行し、流れを検証するためのものです。含まれる内容：

- 事前学習済み speaker encoder を凍結；
- 192 次元埋め込みの後段に residual adapter：`192 → 384 → 192`、ReLU、Dropout 0.40、fc2 ゼロ初期化、`alpha` 初期値 0.1；
- 学習損失：supervised contrastive loss + 0.05 × identity MSE；
- P=8 / K=4 の PK サンプリング、1 epoch あたり 100 batch；
- target-domain embedding の次元ごと標準化と L2 正規化；
- 固定重み 0.5/0.5 のスコア融合；
- AFSC 特徴量はさらに target-domain gap 更新に対応（デモではデフォルト 2 epoch）。

すでに実行済みの第一部成果物に依存します：

```text
runs/demo/last.pt
demo_data/train.csv
demo_data/eval.csv
```

デモを実行：

```bash
python -m Adaptation.demo --device cpu
```

AFSC gap 更新を行わず adapter のみ学習する場合：

```bash
python -m Adaptation.train --device cpu --gap-epochs 0
```

完全な流れ（AFSC はデフォルトで 2 epoch の gap 更新）：

```bash
python -m Adaptation.train --device cpu
```

出力は `runs/demo_adapt/`（または `runs/demo_train/`）：

| ファイル | 内容 |
|---|---|
| `summary.json` | baseline / standardized / adapter / fusion の EER、MinDCF |
| `history.json` | adapter の各 epoch の loss / supcon / identity / alpha |
| `gap_history.json` | `--gap-epochs > 0` のときのみ生成。AFSC gap 更新の過程を記録 |

`summary.json` の四つのブロックの意味：

| フィールド | 意味 | 論文中の列 |
|---|---|---|
| `baseline` | 未適応。事前学習埋め込みのコサインスコア | Before |
| `standardized` | target-domain 次元ごと標準化 + L2 後のスコア | Standardized |
| `adapter` | residual adapter 学習後のスコア | Adapter |
| `fusion` | `0.5 × adapter + 0.5 × standardized` | Fusion |

デモは小型モデル（`Model/demo.yaml`）を使用するため、論文の数値を再現するものではありません。実際の SITW クロスドメイン実験には `Adaptation/run.py`（公開版は未整備）と完全な SITW Dev / Eval プロトコルが必要です。

## 事前学習モデルの読み込み

論文著者は CN-Celeb2 で学習した AFSC + ECAPA 重みを提供しています：

```
PreTrained/embedding_model.ckpt    # AFSC + ECAPA の重み
PreTrained/classifier.ckpt         # 2793 × 192 の分類ヘッド
```

これらは旧形式のため、まず `tools/convert_legacy.py` で本プロジェクトの `format_version: 1` に変換します：

```bash
python tools/convert_legacy.py
```

変換スクリプトは以下を行います：

1. `afsc.*` プレフィックスを `frontend.*` に変更；
2. `ecapa.*` プレフィックスを `encoder.*` に変更；
3. すべての key が `SpeakerSystem` に一致することを検証；
4. `PreTrained/cn_celeb2_afsc_ecapa.pt` と `PreTrained/cn_celeb2_afsc_ecapa.json` を書き出す。

`All keys matched exactly.` と表示されれば変換は完全に成功しています。

### 事前学習モデルで抽出・スコアリング

切り替え方法：`Inference/extract.py`、`Evaluation/score.py`、`Features/export.py`、`Inference/verify.py` を開き、`default_checkpoint()`（または `default_paths()`）の (A) をコメントアウトし、(B) をアンコメントします。

その後は「自分で学習」と同じコマンドが使用できます：

```bash
python -m Inference.extract --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --csv demo_data/eval.csv --output runs/pretrained_eval.npz --device cpu
python -m Evaluation.score --enrol runs/pretrained_eval.npz --trials demo_data/trials.txt --output runs/pretrained_scores
python -m Features.export --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --output runs/pretrained_filters --plot
```

### 事前学習モデルの注意点

- `Speech_example/` の 3 話者は CN-Celeb2 の 2793 話者に**含まれていません**。事前学習モデルは彼らを見たことがないため、このデモでの EER が高くなるのは正常です。
- 意味のある評価には CN-Celeb2 または SITW の公式評価プロトコルと trial リストが必要です。
- 事前学習モデルは AFSC フロントエンドで学習されているため、`feature: afsc` でのみ使用できます。
- チェックポイントに保存された話者マッピングはプレースホルダー（`speaker_00000`…`speaker_02792`）であり、分類結果を実際の話者名に戻す場合にのみ影響します。

## 実データへの切り替え

16 kHz の音声と、学習セット用の wav.scp、utt2spk を用意します。wav.scp は各行が「音声 ID パス」、utt2spk は各行が「音声 ID 話者 ID」です。音声ファイルはリポジトリにコピーしません。

```bash
python -m Dataset.prepare --wav-scp /data/train/wav.scp --utt2spk /data/train/utt2spk --output /data/train/train.csv
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --device cuda
```

GPU がない場合は `--device cpu` も使用できますが、正式なモデル学習は非常に遅くなります。Windows でマルチプロセスの問題が発生する場合は、YAML の `num_workers` を 0 に設定してください。

テストセットの抽出には話者ラベルは不要です：

```bash
python -m Dataset.prepare --wav-scp /data/eval/wav.scp --output /data/eval/eval.csv
python -m Inference.extract --checkpoint runs/afsc/last.pt --csv /data/eval/eval.csv --output runs/afsc/eval.npz --device cuda
python -m Evaluation.score --enrol runs/afsc/eval.npz --trials /data/eval/trials --output runs/afsc/scores
```

trials の各行は「登録音声 ID テスト音声 ID ラベル」で、同一話者は 1、異なる話者は 0 です。実際のデータセットプロトコルに合った分割と trials を使用してください。コードは CN-Celeb の登録集約方法を勝手に推測せず、公式プロトコルに代わるランダムな研究用テストセットも生成しません。

学習コマンドに `--feature fbank` または `--feature mfcc` を追加するとベースラインを切り替えられます。同時に出力ディレクトリも変更してください。抽出時にはチェックポイントから特徴タイプが自動的に読み込まれるため、再度指定する必要はありません。

クロスドメイン適応部分も同じ CSV 形式を受け付けます。`Adaptation.train` と `Adaptation.demo` は `--train-csv` / `--eval-csv` でマニフェストを読み込み、`--checkpoint` で事前学習モデルを指定し、`--chunk-seconds` で長音声処理を制御します（`None` は全体処理、30.0 は 30 秒チャンク）。

## 既存コードとの違い

- `Model/config.yaml`：論文設定。2+8 epoch、margin は 0.3 固定、`channels=[1024,1024,1024,1024,3072]`、埋め込み次元 192。
- `Model/legacy_schedule.yaml`：ログに対応する学習スケジュールオプションを保持。2+10 epoch、margin が 0 から 0.3 へ増加、学習率は第 10 epoch 末に最小値へ。学習スケジュールを表現できるだけで、特定の論文結果に対応することを証明するものではありません。
- `Model/demo.yaml`：縮小モデル。CPU での迅速な検証専用。
- AFSC は元コードの初期化参照配列、正ギャップ正規化、フィルタ式を保持しています。その正確な境界は約 20.039–7614.844 Hz で、最初の実出力点も正規化の影響を受けます。新しい「正確な 20–7600 Hz 初期化」に黙って置き換えてはいません。
- 新しい CSV はデフォルトで 1 発話 1 行とし、学習時にランダムに 3 秒を切り出します。ユーザーが start/stop を明示した場合は、その区間を先に読み込んでから切り出します。元コードの「CSV 複数行セグメントだが学習が区間を無視する」挙動は踏襲していません。
- 新しい学習器は再開用に完全な状態を保存し、新しいチェックポイント形式を採用しています。旧 `embedding_model.ckpt` は直接読み込めず、`tools/convert_legacy.py` で変換が必要です。
- スコアリングは同点スコアと ROC 端点を明示的に扱うため、同点時に旧指標関数と细微な差が出る可能性があります。
- `Adaptation/` の residual adapter は論文 3.5.2 節に従っています。identity loss は detached base embedding に対して計算し、AFSC gap 更新時にフィルタを identity 方向へ引っ張らないようにしています。
- AFSC gap 更新段階では encoder を `eval()` に保ち、`raw_gaps` と adapter のみを更新します。Res2Net と X-Vector は論文では追加の gap 更新を行いません。

## 学習の再開と結果確認

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
python -m Features.export --checkpoint runs/afsc/last.pt --output runs/afsc/filters --plot
```

学習出力には各 epoch のログ、設定、話者番号、完全なチェックポイント、周波数点が含まれます。再開は次の epoch からで、epoch 途中からの再開には対応していません。完全なチェックポイントにはデータマニフェストのパスが含まれるため、モデルを公開する前に確認してください。

## チェックプログラム

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

現在のテスト範囲は TESTING.md を参照してください。テストが通ることはソフトウェアの流れが動作することを意味し、実データセットでの学習と実験検証の代わりにはなりません。

## 帰属

`THIRD_PARTY_NOTICES.md` と `LICENSE` を参照してください。この release は、ユーザー提供の 3D-Speaker / SpeechBrain 派生 ECAPA と前処理コードの上に構築されており、帰属は保持されています。論文で提案された特徴手法を使用する場合は、AFSC 論文を引用してください。研究用音声は元の提供元から取得する必要があります。

`tools/convert_legacy.py` は本プロジェクトのオリジナルコードで、3D-Speaker 形式の AFSC + ECAPA チェックポイントを本プロジェクトの形式に変換するものです。変換後のチェックポイントの内容は元のライセンスに依然として従います。事前学習重みを使用する際は、再配布権があることを確認してください。

`Adaptation/` の residual adapter、supervised contrastive loss、target-domain embedding 標準化、固定スコア融合は、論文 3.5 節と著者提供の実験ノートを参考にしています。Colab 上の SITW 完全実験コードは `speakerlab` と Google Drive に依存しており、本公開リポジトリには含まれていません。

### デモデータセット

`Speech_example/` のデモ音声は CN-Celeb データセット（OpenSLR SLR82）から取得したもので、流れをデモするために使用しており、完全なデータセットを再配布するものではありません。

- データセットホームページ：https://openslr.org/82/
- ライセンス：Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- 出典：清華大学 CSLT（Center for Speech and Language Technologies）が公開

CN-Celeb の引用：

```bibtex
@inproceedings{fan2020cn,
  title={CN-CELEB: a challenging Chinese speaker recognition dataset},
  author={Fan, Yue and Kang, JW and Li, LT and Li, KC and Chen, HL and
          Cheng, ST and Zhang, PY and Zhou, ZY and Cai, YQ and Wang, Dong},
  booktitle={ICASSP 2020-2020 IEEE International Conference on
             Acoustics, Speech and Signal Processing (ICASSP)},
  year={2020}
}
```

本リポジトリのデモ音声またはコードを使用する際は、CC BY-SA 4.0 を遵守し、上記の帰属を保持してください。本リポジトリは完全な CN-Celeb データセットを配布せず、少数のデモ音声のみを含みます。完全なデータセットが必要な場合は https://openslr.org/82/ から取得してください。
