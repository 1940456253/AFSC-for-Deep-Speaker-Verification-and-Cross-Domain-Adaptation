<a id="top"></a>

# 🎙️ AFSC：深層学習による話者照合と領域適応のための音声特徴量

[English](README.md) | [中文](README_中文.md) | **日本語**

**学習可能な音響特徴量 · 話者照合 · 対象領域への適応**

本研究では、AFSC 特徴量と ECAPA-TDNN の学習、埋め込み表現の抽出、話者照合、および領域適応の実装を公開しています。比較用に MFCC と FBank 特徴量も用意しています。Res2Net と X-Vector は元の参照実装を提供しており、必要に応じて各自で組み込むことができます。

> 📢 **公開内容の更新**
>
> - **領域適応の実験コードを公開**：SITW の対象領域への適応に関する実装を [`Adaptation/`](Adaptation/) に公開しました。この実装を用いて追加学習や比較実験を行えます。
> - **事前学習済みモデルを公開**：**CN-Celeb2 で 10 epoch 学習した ECAPA + AFSC** を本研究の [GitHub Releases](../../releases) から取得できます。

| 利用方法 | 実行箇所 | 用途 |
|---|---|---|
| 🏋️ 最初から学習 | `Model/train.py` | 音声例または独自のデータセットで AFSC + ECAPA を学習 |
| 📦 事前学習済みモデルを読み込む | [Releases](../../releases) · `Inference/` | 埋め込み表現の抽出、照合値の計算、2 つの音声の比較 |
| 🔄 対象領域への適応 | [`Adaptation/`](Adaptation/) | 事前学習済みモデルを用いた領域適応の追加学習と照合値の比較 |

本実装で学習したモデルと事前学習済みモデルは、`Inference/`、`Evaluation/`、`Features/` の実行コードを共用します。領域適応には `Adaptation/` を使用します。

### 🧭 目次

[構成](#structure) · [環境構築](#installation) · [実行例のデータ](#demo) · [学習と照合](#training) · [領域適応](#adaptation) · [事前学習済みモデル](#pretrained) · [独自データ](#data) · [設定](#configuration) · [学習の再開](#resume) · [動作確認](#tests) · [引用と帰属表示](#credits)

---

<a id="structure"></a>

## 📂 構成

| Directory | 主なファイル | 機能 |
|---|---|---|
| Dataset | data.py、prepare.py、demo.py | 音声の読み込み、学習用一覧の作成、話者別の保存先から実行例のデータを生成 |
| Features | spectrum.py、afsc.py、frontend.py、export.py | Power spectrum と 3 種類の音響特徴量、学習済み filter の出力 |
| Model | ecapa_tdnn.py、system.py、train.py | ECAPA、分類損失、2 段階の学習と再開 |
| Evaluation | metrics.py、score.py、test_pipeline.py、smoke_test.py | 照合値の計算、EER/MinDCF、動作確認用の実装 |
| Inference | extract.py、verify.py | 埋め込み表現の抽出、2 つの音声の比較 |
| Adaptation | adapter.py、losses.py、chunking.py、standardize.py、trials.py、sampler.py、dataset.py、embed.py、train.py、demo.py | 対象領域への適応：residual adapter、embedding の標準化、固定重みでの統合、AFSC gap の更新 |

その他の保存先：

| Directory | 内容 |
|---|---|
| `Speech_example/` | 音声例：話者ごとに保存先を分け、それぞれ 15 発話を収録 |
| `PreTrained/` | Releases から取得した事前学習済み重みと、変換後の `cn_celeb2_afsc_ecapa.pt` の保存先 |
| `tools/` | `convert_legacy.py`、`inspect_ckpt.py` などの補助コード |

学習用の実装は Model に含めているため、Training という別の保存先は設けていません。すべての実行命令は repository の最上位で実行してください。

<a id="installation"></a>

## 🛠️ 環境構築

Python 3.10–3.12 を使用します。まず PyTorch と torchaudio を対応する 2.5.1 で導入し、その後 requirements.txt の依存関係を導入してください。CPU 版の例：

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

GPU を使用する場合は、対応する CUDA 版を導入してください。`requirements.txt` の変更は不要です。

<a id="demo"></a>

## 🚀 実行例のデータを準備

`Speech_example/` から一覧を生成します。

```bash
python -m Dataset.demo
```

出力例：

```
Train:  30 utterances from 3 speakers
Eval:   15 utterances from 3 speakers
Trials: 105 (30 target, 75 nontarget)
Manifests written to .../demo_data
```

生成したファイルは `demo_data/` に保存されます。

| ファイル | 内容 |
|---|---|
| `train.csv` | 学習用一覧。各行は `ID,path,spk` |
| `eval.csv` | 評価用一覧。各行は `ID,path,spk` |
| `trials.txt` | 評価用の照合対。各行は `enrol_id test_id label` |

既定では、各話者の 5 発話を評価用に取り分け、残りを学習に使用します。変更する場合：

```bash
python -m Dataset.demo --eval-per-speaker 3
```

<a id="training"></a>

## 🏋️ 第 1 部：学習と話者照合

PyCharm で repository の最上位を開き、依存関係を導入済みの Python interpreter を選択します。実行設定の作業場所も repository の最上位にしてください。各実装の既定の実行例設定を使って順番に実行するか、以下の命令を端末で実行できます。

### 1. 学習

```bash
python -m Model.train --config Model/demo.yaml --train-csv demo_data/train.csv --output runs/demo --device cpu
```

`Model/demo.yaml` は CPU でも動かしやすい小規模な構成です。`Model/config.yaml` は論文の設定（2+8 epoch、`channels=[1024,1024,1024,1024,3072]`、192 次元の埋め込み表現）であり、本実験にはこちらを使用します。

学習結果は `runs/demo/` に保存されます。

| ファイル | 内容 |
|---|---|
| `config.yaml` | 今回の学習に使用した全設定 |
| `speakers.json` | 話者 → 番号の対応表 |
| `train.jsonl` | 各 epoch の loss / accuracy / lr / margin |
| `epoch_XXX.pt` | 各 epoch の完全な checkpoint |
| `last.pt` | 最終 epoch の checkpoint の複製 |
| `frequency_points_XXX.json` | 各 epoch の AFSC 周波数点 |

### 2. 評価用の埋め込み表現を抽出

```bash
python -m Inference.extract --checkpoint runs/demo/last.pt --csv demo_data/eval.csv --output runs/demo/embeddings.npz --device cpu
```

出力：

- `runs/demo/embeddings.npz`：`ids`（文字列配列）と `embeddings`（N × D）を格納
- `runs/demo/embeddings.json`：checkpoint、csv、feature、utterances、chunk_seconds を記録した付随情報

### 3. 照合値を計算

```bash
python -m Evaluation.score --enrol runs/demo/embeddings.npz --trials demo_data/trials.txt --output runs/demo/scores
```

出力：

- `runs/demo/scores/scores.csv`：各照合対の enrol_id、test_id、label、score
- `runs/demo/scores/metrics.json`：EER、MinDCF、照合対の数

EER は百分率、MinDCF は正規化した値で示します。Target の事前確率の既定値は `0.01` です。

### 4. Filter を出力（任意）

```bash
python -m Features.export --checkpoint runs/demo/last.pt --output runs/demo/filters --plot
```

出力：

- `runs/demo/filters/afsc_filters.npz`：`bin_points` (82,)、`hz` (82,)、`filters` (80, 257) を格納
- `runs/demo/filters/afsc_filters.png`（`--plot` を指定した場合）

### 5. 2 つの音声を比較（任意）

```bash
python -m Inference.verify --checkpoint runs/demo/last.pt --enrol Speech_example/speech_demo1/speech-01-001.flac --test Speech_example/speech_demo1/speech-01-002.flac
```

`--enrol` / `--test` を省略すると、`Speech_example/speech_demo1/` 内の最初の 2 つの音声を使用します。`--threshold` は開発用データで調整する必要があり、すべての条件に共通する既定値はありません。

<a id="adaptation"></a>

## 🔄 第 2 部：領域適応と追加学習

**SITW の領域適応実験の実装は [`Adaptation/`](Adaptation/) に公開しています。** 公開コード、事前学習済みモデル、および独自の対象領域のデータを用いて追加学習を行い、学習 epoch 数、標本抽出の設定、AFSC gap の更新方針を調整できます。

この部分は論文の 3.5 節に対応します。以下の命令では、まず `demo_data/` の 3 話者で処理手順を確認します。本実験では SITW Dev / Eval の完全な分割と、対応する公式の trials を使用してください。

### 手法の構成

論文実験の主な設定は次のとおりです。

- 事前学習済み speaker encoder を固定；
- 論文のモデルが出力する 192 次元の埋め込み表現の後に residual adapter を接続：`192 → 384 → 192`、ReLU、Dropout 0.40、fc2 はゼロで初期化、`alpha` の初期値は 0.1；
- 学習損失：supervised contrastive loss + 0.05 × identity MSE；
- 論文実験では P=8 / K=4 の PK sampling を使用し、各 epoch は 100 batch；小規模な実行例では、実際の話者数に合わせた標本抽出の設定が必要；
- 対象領域の embedding を次元ごとに標準化した後、L2 正規化；
- 固定重み 0.5/0.5 で照合値を統合；
- AFSC 特徴量では、対象領域での gap 更新にも対応（実行例の既定値は 2 epoch）。

### 実行例

以下の実行例では、第 1 部で生成したファイルを使用します。

```text
runs/demo/last.pt
demo_data/train.csv
demo_data/eval.csv
```

実行例を動かします。

```bash
python -m Adaptation.demo --device cpu
```

AFSC gap を更新せず、adapter のみを学習する場合：

```bash
python -m Adaptation.train --device cpu --gap-epochs 0
```

全体の処理（AFSC gap 更新の既定値は 2 epoch）：

```bash
python -m Adaptation.train --device cpu
```

出力先は `runs/demo_adapt/`（または `runs/demo_train/`）です。

| ファイル | 内容 |
|---|---|
| `summary.json` | baseline / standardized / adapter / fusion の 4 種類の照合値に対する EER、MinDCF |
| `history.json` | 各 epoch の adapter の loss / supcon / identity / alpha |
| `gap_history.json` | `--gap-epochs > 0` の場合のみ生成。AFSC gap の更新過程を記録 |

`summary.json` の 4 項目の意味：

| 項目 | 意味 | 論文の対応する列 |
|---|---|---|
| `baseline` | 適応前の事前学習済み embedding から直接求めた cosine 類似度 | Before |
| `standardized` | 対象領域で次元ごとに標準化し、L2 正規化した後の照合値 | Standardized |
| `adapter` | residual adapter 側の照合値。gap 更新を有効にした場合は、特徴抽出部の適応の影響も含む | Adapter |
| `fusion` | `0.5 × adapter + 0.5 × standardized` | Fusion |

> 💡 **実行例と本実験**
>
> 実行例では小規模な構成（`Model/demo.yaml`）と少量の音声を使用するため、その結果は論文の数値を示すものではありません。本実験では Releases の CN-Celeb2 事前学習済みモデルを利用し、`Adaptation/` の実装に従って SITW のデータ保存先、学習設定、評価手順を指定してください。

### SITW または独自の対象領域で実験

1. 事前学習済み重みを取得し、次節に従って本実装で読み込める checkpoint を準備します。
2. 対象領域の学習用・評価用一覧と、評価手順に合った trials を準備します。
3. `Adaptation/` でモデルとデータの保存先、および追加学習の設定を指定します。
4. 適応前、標準化、adapter、統合した照合値の EER / MinDCF を比較します。

SITW 実験では Dev を適応と設定値の選択に使用し、Eval を最終評価に使用します。標準化に必要な統計量は対象領域の適応用データから推定してください。音声例は実行方法を理解するためのものであり、正式な領域適応の評価手順に代わるものではありません。

<a id="pretrained"></a>

## 📦 事前学習済みモデルの取得と読み込み

### ⬇️ モデルの取得

**CN-Celeb2 で 10 epoch 学習した ECAPA + AFSC の事前学習済みモデルを公開しています。本研究の [Releases](../../releases) から取得できます。**

Releases を開き、該当する版の **Assets** から重みファイルまたは圧縮ファイルを取得します。圧縮されている場合は展開し、repository 最上位の `PreTrained/` に配置してください。

| 項目 | 説明 |
|---|---|
| モデル | ECAPA-TDNN + AFSC |
| 事前学習用データ | CN-Celeb2 |
| 学習回数 | 10 epoch |
| 埋め込み表現の次元数 | 192 |
| 取得先 | [本研究の GitHub Releases](../../releases) |
| 保存先 | `PreTrained/` |

### 旧形式の重みの変換

旧形式の checkpoint を取得した場合は、以下のように配置します。

```
PreTrained/embedding_model.ckpt    # AFSC + ECAPA の重み
PreTrained/classifier.ckpt         # 2793 × 192 の分類層
```

旧形式のファイルは、まず `tools/convert_legacy.py` で本実装の `format_version: 1` に変換します。

```bash
python tools/convert_legacy.py
```

変換処理の内容：

1. `afsc.*` の接頭辞を `frontend.*` に変更；
2. `ecapa.*` の接頭辞を `encoder.*` に変更；
3. すべての key が `SpeakerSystem` に一致することを確認；
4. `PreTrained/cn_celeb2_afsc_ecapa.pt` と `PreTrained/cn_celeb2_afsc_ecapa.json` を出力。

`All keys matched exactly.` と表示されれば、重みの key が一致しています。

> 💡 取得したファイルに変換済みの `cn_celeb2_afsc_ecapa.pt` が含まれている場合は、そのまま使用でき、再変換は不要です。

### 事前学習済みモデルによる抽出と照合

**端末から実行する場合：**引数でモデルと出力先を直接指定できるため、実装内の既定値を変更する必要はありません。

```bash
python -m Inference.extract --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --csv demo_data/eval.csv --output runs/pretrained_eval.npz --device cpu
python -m Evaluation.score --enrol runs/pretrained_eval.npz --trials demo_data/trials.txt --output runs/pretrained_scores
python -m Features.export --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --output runs/pretrained_filters --plot
```

PyCharm から引数なしで直接実行する場合は、各実装の (A)/(B) の注記に従って `default_checkpoint()` または `default_paths()` を切り替え、モデル、埋め込み表現、出力の保存先が対応するようにしてください。

### 利用上の説明

- 小規模な実行例の EER は、標本数や音声条件の影響を受けやすく、事前学習済みモデルの総合的な性能を判断する根拠にはなりません。話者照合は本来、学習時に含まれない話者を対象とするため、「学習時にその話者を見ていない」という理由だけで評価誤差を説明するべきではありません。
- 意味のある評価には、CN-Celeb2 または SITW の公式評価手順と照合対の一覧を使用してください。
- 事前学習済みモデルは AFSC を特徴抽出部に用いて学習しているため、`feature: afsc` のみで使用できます。
- checkpoint 内の話者対応表には仮の名称（`speaker_00000`…`speaker_02792`）を使用しています。これは分類結果を実際の話者名に対応付ける際にのみ影響します。

<a id="data"></a>

## 🗂️ 独自のデータセットを使用

16 kHz の音声と、学習用の wav.scp、utt2spk を準備します。wav.scp の各行は「音声 ID 保存先」、utt2spk の各行は「音声 ID 話者 ID」です。音声ファイルは repository 内に複製しません。

```bash
python -m Dataset.prepare --wav-scp /data/train/wav.scp --utt2spk /data/train/utt2spk --output /data/train/train.csv
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --device cuda
```

GPU がない場合は `--device cpu` を使用できますが、本実験用の構成では学習に長時間かかります。Windows で複数 process の処理に問題が生じた場合は、YAML の `num_workers` を 0 に設定してください。

評価用の埋め込み表現の抽出には、話者ラベルは不要です。

```bash
python -m Dataset.prepare --wav-scp /data/eval/wav.scp --output /data/eval/eval.csv
python -m Inference.extract --checkpoint runs/afsc/last.pt --csv /data/eval/eval.csv --output runs/afsc/eval.npz --device cuda
python -m Evaluation.score --enrol runs/afsc/eval.npz --trials /data/eval/trials --output runs/afsc/scores
```

trials の各行は「登録音声 ID 評価音声 ID 正解ラベル」で、同一話者を 1、異なる話者を 0 とします。実際のデータセットの評価手順に合った分割と trials を使用してください。本実装は CN-Celeb の登録音声の集約方法を自動で推測せず、公式の評価手順の代わりとなる無作為な研究用評価集合も生成しません。

学習命令に `--feature fbank` または `--feature mfcc` を追加すると比較手法を切り替えられます。出力先も変更してください。抽出時は checkpoint から特徴量の種類を自動で読み取るため、再指定は不要です。

領域適応も同じ CSV 形式に対応しています。`Adaptation.train` と `Adaptation.demo` は `--train-csv` / `--eval-csv` で一覧を読み込み、`--checkpoint` で事前学習済みモデルを指定します。長い音声の処理は `--chunk-seconds` で設定します（`None` は全体を一度に処理、30.0 は 30 秒の区間に分割して処理）。

<a id="configuration"></a>

## ⚙️ 設定と実装の説明

- `Model/config.yaml`：論文の設定。2+8 epoch、margin は 0.3 固定、`channels=[1024,1024,1024,1024,3072]`、埋め込み表現は 192 次元。
- `Model/legacy_schedule.yaml`：記録に対応した学習計画の選択肢を保存しています。2+10 epoch、margin は 0 から 0.3 に増加し、学習率は第 10 epoch の終了時に最小値となります。これは学習計画を表すものであり、論文の特定の結果との対応を証明するものではありません。
- `Model/demo.yaml`：CPU で短時間に動作確認するための小規模な構成。
- AFSC は元の実装の初期化参照配列、正の間隔の正規化、filter の計算式を維持しています。境界は約 20.039–7614.844 Hz であり、最初に実際に出力される周波数点も正規化の影響を受けます。正確な 20–7600 Hz に再設定せず、この初期化を引き継いでいます。
- 新しい CSV は、既定では 1 音声につき 1 行とし、学習時に 3 秒間を無作為に切り出します。start/stop が明示されている場合は、まずその区間を読み込んでから切り出します。元の実装にあった「CSV では複数行に区間分割するが、学習時には区間を無視する」動作は引き継いでいません。
- 新しい学習処理は再開に必要な状態をすべて保存し、新しい checkpoint 形式を使用します。旧形式の `embedding_model.ckpt` は直接読み込めないため、先に `tools/convert_legacy.py` で変換してください。
- 照合値が同値の場合と ROC の端点を明示的に処理するため、同値がある場合には旧評価関数とわずかな差が生じることがあります。
- `Adaptation/` の residual adapter は論文の 3.5.2 節を参照しています。Identity loss では detached base embedding を参照対象とし、参照対象側には勾配を伝えません。
- AFSC gap の更新中は encoder を `eval()` に保ち、`raw_gaps` と adapter のみを更新します。論文では Res2Net と X-Vector に追加の gap 更新を行っていません。

<a id="resume"></a>

## 💾 学習の再開と結果の確認

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
python -m Features.export --checkpoint runs/afsc/last.pt --output runs/afsc/filters --plot
```

学習結果には、各 epoch の記録、設定、話者番号、完全な checkpoint、周波数点が含まれます。学習は次の epoch から再開し、epoch の途中からの再開には対応していません。完全な checkpoint にはデータ一覧の保存先が含まれるため、モデルを公開する前に確認してください。

<a id="tests"></a>

## 🧪 動作確認

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

現在の確認範囲は TESTING.md を参照してください。確認に通ることは一連の処理が実行できることを示しますが、実際のデータセットによる学習や実験的な検証に代わるものではありません。

<a id="credits"></a>

## 📚 引用と帰属表示

実装の出典と利用許諾は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) と [`LICENSE`](LICENSE) を参照してください。本実装は 3D-Speaker / SpeechBrain に由来する ECAPA と前処理のコードを基に整理し、関連する帰属表示を保持しています。提案した特徴量の手法を使用する場合は、AFSC 論文を引用してください。研究用音声は元の提供元から取得する必要があります。

`tools/convert_legacy.py` は、3D-Speaker 形式の AFSC + ECAPA checkpoint を本実装の形式に変換するために独自に作成したコードです。変換後の checkpoint の内容にも元の利用許諾が適用されます。事前学習済み重みを使用する際は、再配布の権限を確認してください。

`Adaptation/` の residual adapter、supervised contrastive loss、対象領域の embedding の標準化、および固定重みによる照合値の統合は、論文の 3.5 節と著者の実験記録を参照しています。対応する SITW 領域適応実験のコードを `Adaptation/` に公開しており、データを設定して追加学習実験を行えます。

### 実行例のデータセット

`Speech_example/` には CN-Celeb（OpenSLR SLR82）からの少量の音声が含まれ、一連の処理の実行例としてのみ使用します。本 repository に完全なデータセットは含まれていません。

- データセットの公開元：https://openslr.org/82/
- 利用許諾：Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- 出典：清華大学 CSLT（Center for Speech and Language Technologies）が公開

CN-Celeb：

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

上記の音声例を使用する際は、CC BY-SA 4.0 に従い、帰属表示を保持してください。コードの利用許諾は、本 repository の `LICENSE` と `THIRD_PARTY_NOTICES.md` を参照してください。本 repository は少量の音声例のみを含み、完全な CN-Celeb データセットは配布していません。完全なデータセットは https://openslr.org/82/ から取得してください。


---

[⬆️ 先頭へ戻る](#top)
