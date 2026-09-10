
# AFSC + ECAPA 第一部分公開コード

[English](README.md) | [中文](README_中文.md) | **日本語**

このプロジェクトは、論文の第一部分である AFSC 特徴、ECAPA 学習、埋め込み抽出、話者照合を扱い、MFCC、FBank の比較も含みます。クロスドメイン適応は含まれず、Res2Net や X-Vector も含まれません。コードは再学習可能で、論文が提供する CN-Celeb2 事前学習モデルの読み込みにも対応しています。

プロジェクトは二通りの使い方をサポートします：

- **自分で学習**：`Speech_example/` または独自データセットから出発し、`Model/train.py` で AFSC + ECAPA を学習；
- **事前学習を読み込む**：論文が提供する `embedding_model.ckpt` + `classifier.ckpt` を `tools/convert_legacy.py` で本プロジェクトの checkpoint 形式に変換し、そのまま抽出・スコアリング・比較に使う。

両方式は同じ `Inference/`、`Evaluation/`、`Features/` スクリプトを共有します。

## 主なディレクトリ

| フォルダ | 主なファイル | 機能 |
|---|---|---|
| Dataset | data.py、prepare.py、demo.py | 音声読み込み、学習マニフェスト作成、話者フォルダからの demo データ生成 |
| Features | spectrum.py、afsc.py、frontend.py、export.py | パワースペクトルと三種の音響特徴、学習後のフィルタ出力 |
| Model | ecapa_tdnn.py、system.py、train.py | ECAPA、分類損失、二段階学習と再開 |
| Evaluation | metrics.py、score.py、test_pipeline.py、smoke_test.py | スコアリング、EER/MinDCF、検証コード |
| Inference | extract.py、verify.py | 埋め込み抽出、二波形比較 |

さらに以下があります：

| フォルダ | 内容 |
|---|---|
| `Speech_example/` | サンプル音声。各サブフォルダが一人の話者、各フォルダに 15 発話 |
| `PreTrained/` | 論文提供の事前学習 checkpoint（`embedding_model.ckpt` + `classifier.ckpt`）と変換後の `cn_celeb2_afsc_ecapa.pt` |
| `tools/` | 補助スクリプト。`convert_legacy.py` と `inspect_ckpt.py` を含む |

学習コードは Model に配置しているため、第六の Training ディレクトリは追加していません。すべてのコマンドはプロジェクトルートから実行します。

## インストール

Python 3.10–3.12 を使用してください。対応する PyTorch と torchaudio 2.5.1 をインストールしてから、requirements.txt をインストールします。CPU 版の例：

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

生成されるファイルは `demo_data/` に：

| ファイル | 内容 |
|---|---|
| `train.csv` | 学習マニフェスト。各行は `ID,path,spk` |
| `eval.csv` | 評価マニフェスト。各行は `ID,path,spk` |
| `trials.txt` | 評価試行。各行は `enrol_id test_id label` |

デフォルトでは話者ごとに 5 件を評価用に確保し、残りを学習用にします。変更も可能です：

```bash
python -m Dataset.demo --eval-per-speaker 3
```

## 自分で学習する完全な流れ

PyCharm で以下のファイルを順に右クリック Run するか、コマンドラインから `python -m ...` で順に実行します。

### 1. 学習

```bash
python -m Model.train --config Model/demo.yaml --train-csv demo_data/train.csv --output runs/demo --device cpu
```

`Model/demo.yaml` は CPU 向けの小さいモデルです。`Model/config.yaml` は論文の設定（2+8 epoch、`channels=[1024,1024,1024,1024,3072]`、埋め込み次元 192）で、正式な実験にはこちらを使います。

学習出力は `runs/demo/` に：

| ファイル | 内容 |
|---|---|
| `config.yaml` | 今回使用した完全な設定 |
| `speakers.json` | 話者 → インデックスのマッピング |
| `train.jsonl` | 各 epoch の loss / accuracy / lr / margin |
| `epoch_XXX.pt` | 各 epoch の完全な checkpoint |
| `last.pt` | 最後の epoch のコピー |
| `frequency_points_XXX.json` | 各 epoch の AFSC 周波数点 |

### 2. 評価セットの埋め込み抽出

```bash
python -m Inference.extract --checkpoint runs/demo/last.pt --csv demo_data/eval.csv --output runs/demo/embeddings.npz --device cpu
```

出力：

- `runs/demo/embeddings.npz`：`ids`（文字列配列）と `embeddings`（N × D）を含む
- `runs/demo/embeddings.json`：メタデータ。checkpoint、csv、feature、utterances、chunk_seconds を記録

### 3. スコアリング

```bash
python -m Evaluation.score --enrol runs/demo/embeddings.npz --trials demo_data/trials.txt --output runs/demo/scores
```

出力：

- `runs/demo/scores/scores.csv`：各試行の enrol_id、test_id、label、score
- `runs/demo/scores/metrics.json`：EER、MinDCF、試行数

EER の単位はパーセント、MinDCF は正規化値で、デフォルトの目標事前確率は `0.01` です。

### 4. フィルタ出力（任意）

```bash
python -m Features.export --checkpoint runs/demo/last.pt --output runs/demo/filters --plot
```

出力：

- `runs/demo/filters/afsc_filters.npz`：`bin_points` (82,)、`hz` (82,)、`filters` (80, 257) を含む
- `runs/demo/filters/afsc_filters.png`（`--plot` を付けた場合）

### 5. 二波形比較（任意）

```bash
python -m Inference.verify --checkpoint runs/demo/last.pt --enrol Speech_example/speech_demo1/speech-01-001.flac --test Speech_example/speech_demo1/speech-01-002.flac
```

`--enrol` / `--test` を指定しない場合、`Speech_example/speech_demo1/` の先頭二つをデフォルトで使用します。`--threshold` は開発セットで校正する必要があり、普遍的なデフォルト値はありません。

## 事前学習モデルを読み込む

論文著者は CN-Celeb2 で学習した AFSC + ECAPA の重みを提供しています：

```
PreTrained/embedding_model.ckpt    # AFSC + ECAPA の重み
PreTrained/classifier.ckpt         # 2793 × 192 の分類ヘッド
```

これらは旧形式なので、まず `tools/convert_legacy.py` で本プロジェクトの `format_version: 1` に変換します：

```bash
python tools/convert_legacy.py
```

変換スクリプトは次を行います：

1. `afsc.*` プレフィックスを `frontend.*` に変更；
2. `ecapa.*` プレフィックスを `encoder.*` に変更；
3. すべての key が `SpeakerSystem` にマッチするか検証；
4. `PreTrained/cn_celeb2_afsc_ecapa.pt` と `PreTrained/cn_celeb2_afsc_ecapa.json` を出力。

`All keys matched exactly.` と表示されれば、変換は完全に成功しています。

### 事前学習モデルで抽出・スコアリング

切り替え方法：`Inference/extract.py`、`Evaluation/score.py`、`Features/export.py`、`Inference/verify.py` を開き、`default_checkpoint()`（または `default_paths()`）の (A) ブロックをコメントアウトし、(B) ブロックのコメントを外します。

その後は「自分で学習」と同じコマンドで使えます：

```bash
python -m Inference.extract --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --csv demo_data/eval.csv --output runs/pretrained_eval.npz --device cpu
python -m Evaluation.score --enrol runs/pretrained_eval.npz --trials demo_data/trials.txt --output runs/pretrained_scores
python -m Features.export --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --output runs/pretrained_filters --plot
```

### 事前学習モデルを使う際の注意

- `Speech_example/` の 3 人の話者は CN-Celeb2 の 2793 人には**含まれません**。事前学習モデルは彼らを見ていないので、この demo では EER が高めになる可能性があります。これは正常です。
- 本当に意味のある評価は、CN-Celeb2 や SITW の公式評価プロトコルと試行リストで行います。
- 事前学習モデルは AFSC フロントエンドで学習されているため、`feature: afsc` でのみ使用できます。
- checkpoint に保存されている話者マッピングはプレースホルダ（`speaker_00000`…`speaker_02792`）で、分類結果を実話者名に戻す場合にのみ影響します。

## 実データに切り替える

16 kHz の音声、および学習セットの wav.scp、utt2spk を準備します。wav.scp の各行は「音声 ID パス」、utt2spk の各行は「音声 ID 話者 ID」です。音声ファイルはリポジトリにコピーしません。

```bash
python -m Dataset.prepare --wav-scp /data/train/wav.scp --utt2spk /data/train/utt2spk --output /data/train/train.csv
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --device cuda
```

GPU がない場合は `--device cpu` を使用できますが、正式なモデル学習は非常に遅くなります。Windows でマルチプロセスに問題がある場合は、YAML の `num_workers` を 0 にしてください。

テストセットの抽出に話者ラベルは不要です：

```bash
python -m Dataset.prepare --wav-scp /data/eval/wav.scp --output /data/eval/eval.csv
python -m Inference.extract --checkpoint runs/afsc/last.pt --csv /data/eval/eval.csv --output runs/afsc/eval.npz --device cuda
python -m Evaluation.score --enrol runs/afsc/eval.npz --trials /data/eval/trials --output runs/afsc/scores
```

trials の各行は「登録音声 ID テスト音声 ID ラベル」で、同一話者は 1、異なる話者は 0 です。実際のデータセットプロトコルに合った分割と trials を使用してください。コードは CN-Celeb の登録セット集約方法を勝手に推測せず、公式プロトコルに代わるランダム研究試行も生成しません。

学習コマンドに `--feature fbank` または `--feature mfcc` を追加すると、ベースラインに切り替えられます。同時に出力ディレクトリも変えてください。抽出時は checkpoint から特徴タイプを自動的に読み取るので、再指定は不要です。

## 設定と元のコードとの違い

- `Model/config.yaml`：論文の設定。2+8 epoch、margin は 0.3 固定、`channels=[1024,1024,1024,1024,3072]`、埋め込み次元 192。
- `Model/legacy_schedule.yaml`：ログに対応する学習スケジュールのオプション。2+10 epoch、margin を 0 から 0.3 にランプ、学習率は第 10 epoch 末で最小値に達する。学習スケジュールを表現するだけで、ある論文結果に対応することを証明するものではありません。
- `Model/demo.yaml`：縮小モデル。CPU での高速検証専用。
- AFSC は元コードの初期化参照配列、正ギャップ正規化、フィルタ式を保持しています。その正確な境界はおよそ 20.039–7614.844 Hz で、初回の実際の出力点も正規化の影響を受けます。これを「正確な 20–7600 Hz 初期化」に黙って置き換えてはいません。
- 新しい CSV はデフォルトで 1 音声 1 行で、学習時にランダムに 3 秒をクロップします。ユーザーが start/stop を明示した場合は、その区間を読んでからクロップします。元コードの「CSV 複数行セグメントだが学習は区間を無視」という動作は継承していません。
- 新しい学習器は学習を再開できるよう完全な状態を保存し、新しい checkpoint 形式を採用しています。旧 `embedding_model.ckpt` は直接読み込めず、`tools/convert_legacy.py` で変換する必要があります。
- スコアリングは同一スコアと ROC 端点を明示的に扱うため、スコアが同じ場合に旧指標関数とわずかに異なることがあります。

## 学習の再開と結果の確認

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
python -m Features.export --checkpoint runs/afsc/last.pt --output runs/afsc/filters --plot
```

学習出力には、epoch ごとのログ、設定、話者番号、完全な checkpoint、周波数点が含まれます。再開は次の epoch からで、epoch 途中からの再開はサポートしていません。完全な checkpoint にはデータマニフェストのパスが含まれるので、モデルを公開する前にこれらのパスを確認してください。

## プログラムの検証

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

現在のテスト範囲は TESTING.md を参照してください。テストが通ることはソフトウェアフローが動作することを意味し、実データでの学習や実験検証の代わりにはなりません。

## 帰属

`THIRD_PARTY_NOTICES.md` と `LICENSE` を参照してください。このリリースは、ユーザー提供の 3D-Speaker / SpeechBrain 由来の ECAPA と前処理コードに基づいており、帰属は保持されています。論文が提案する特徴手法を使用する場合は、AFSC 論文を引用してください。研究用音声は元の提供元から入手する必要があります。

`tools/convert_legacy.py` は本プロジェクトのオリジナルコードで、3D-Speaker 形式の AFSC + ECAPA checkpoint を本プロジェクト形式に変換します。変換後の checkpoint の内容は元のライセンスに従います。事前学習重みを使用する場合は、再配布権があることを確認してください。
```
