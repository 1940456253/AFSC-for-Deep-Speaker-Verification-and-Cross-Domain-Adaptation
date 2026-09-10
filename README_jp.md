# AFSC + ECAPA-TDNN: 学習と話者照合

**Adaptive Frequency Spectral Coefficients for Deep Speaker Verification: Training and Cross-Domain Adaptation** の第一部のスタンドアロン実装です。

このリリースでは、AFSC フロントエンドと、ECAPA-TDNN の学習、埋め込み抽出、評価パイプラインを提供します。MFCC と FBank は、同じエンコーダでベースラインとして選択できます。Res2Net、X-Vector、ターゲットドメイン適応はこのリリースの対象外です。このリポジトリは研究用データセットや事前学習済み研究チェックポイントを再配布せず、特定されていない過去のチェックポイントや原稿のすべての数値結果を再現することを主張しません。

## 主要ディレクトリ

| ディレクトリ | 内容 |
|---|---|
| `Dataset` | 音声/CSV の読み込み、wav.scp + utt2spk 変換、合成デモ生成器 |
| `Features` | パワースペクトル、MFCC/FBank、学習可能な AFSC とフィルタ出力 |
| `Model` | ECAPA、コサイン分類器、角度マージン損失、二段階学習と YAML レシピ |
| `Evaluation` | 試行スコアリング、EER/MinDCF、単体テストと統合テスト |
| `Inference` | チェックポイント読み込み、発話埋め込み、二波形比較 |

以下のすべてのコマンドは、**このリポジトリのルートディレクトリから**実行します。Windows でも同じ `python -m ...` コマンドが動作します。マルチプロセスが不便な場合は `num_workers: 0` を使用してください。`speakerlab`、Kaldi バイナリ、`kaldiio` のインストールは不要です。固定フロントエンドには、Torchaudio の Kaldi 互換 Python 関数を使用します。

## インストール

Python 3.10～3.12 を使用してください。付属の検証は、CPU 上の Python 3.12 で実行されました。

```bash
python -m venv .venv
```

Linux/macOS では `source .venv/bin/activate`、Windows cmd では `.venv\Scripts\activate` で有効化します。対応する PyTorch と torchaudio のビルドをインストールしてください。CPU の場合:

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

NVIDIA GPU の場合、そのマシンに対応する互換性のある PyTorch/torchaudio 2.5.1 ビルドをインストールし、その後 `requirements.txt` をインストールします。参照モデルは大きいため、迅速な CPU チェックにはデモレシピを使用してください。

## 簡単なエンドツーエンド例

人工トーンと小規模な学習/評価マニフェストを生成します:

```bash
python -m Dataset.demo --output demo_data
python -m Model.train --config Model/demo.yaml --train-csv demo_data/train.csv --output runs/demo --device cpu
python -m Inference.extract --checkpoint runs/demo/last.pt --csv demo_data/eval.csv --output runs/demo/embeddings.npz --device cpu
python -m Evaluation.score --enrol runs/demo/embeddings.npz --trials demo_data/trials.txt --output runs/demo/scores
python -m Features.export --checkpoint runs/demo/last.pt --output runs/demo/filters --plot
```

**デモでは合成トーンと縮小版 ECAPA を使用します。そのスコアはソフトウェアチェックであり、話者照合の研究結果ではありません。** 学習用と評価用の合成 ID は重複していません。実在人物の音声は含まれていません。

## 研究用データの準備

データセットは提供元から入手してください。意図されたデータセットプロトコルに従い、学習用と評価用に別々のマニフェストを用意してください。自動的なランダム train/test 分割や、ランダム生成された研究用試行は行いません。学習/評価間の話者リークを避け、ハイパーパラメータや動作閾値は別の開発データで選択してください。

`wav.scp` 形式（1 行につき 1 発話）:

```text
utt001 /path/to/audio001.wav
utt002 /path/to/audio002.flac
```

スペースを含むパスをサポートします。相対パスは `wav.scp` を基準に解決されます。`sox ... |` のようなシェルパイプラインはサポートされません。`utt2spk` の内容:

```text
utt001 speaker001
utt002 speaker002
```

学習用 CSV を作成します:

```bash
python -m Dataset.prepare --wav-scp /data/train/wav.scp --utt2spk /data/train/utt2spk --output /data/train/train.csv
```

評価用 CSV を作成します（話者ラベルは任意）:

```bash
python -m Dataset.prepare --wav-scp /data/eval/wav.scp --output /data/eval/eval.csv
```

CSV の列は `ID,path,spk,dur` です。抽出には `ID,path` のみが必須で、学習にはさらに `spk` が必要です。CSV の相対パスは CSV の隣を基準に解決されます。音声はすでに **16 kHz** である必要があり、マルチチャネル音声ではチャネル 0 を使用します。各学習行からランダムな 3 秒クロップが 1 つ生成され、短い録音は右側にゼロ埋めされます。デフォルトの変換器は 1 発話につき 1 行を出力します。任意の `start,stop` CSV フィールド（秒単位）は、ランダムクロップ前に反映されます。変換器はこれらのフィールドを生成しません。

評価試行は次の形式で指定します:

```text
utt001 utt002 0
utt001 utt003 1
```

ラベルは `0`/`1` または `nontarget`/`target` にできます。すべての ID は対応する埋め込みアーカイブに含まれている必要があります。1 つの ID は 1 つの発話埋め込みを表します。複数発話の登録集約、VAD、スコア正規化は暗黙には行われません。データセットプロトコルが登録集約や前処理を要求する場合は、評価前にそのプロトコルを実装し文書化してください。

## AFSC + ECAPA の学習

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --device cuda
```

デフォルトのエンコーダはチャネル `[1024,1024,1024,1024,3072]`、80 入力特徴チャネル、192 出力次元です。学習では正規化コサイン分類器と角度マージン損失（`scale=32`、`margin=0.3`）を使用します。AFSC ギャップパラメータは 2 エポックの間、30 倍の学習率乗数と重み減衰ゼロで同時に学習され、その後 8 エポックの間凍結されます。SGD モメンタムはデフォルトで段階遷移時にリセットされ、付属スクリプトのオプティマイザ再構築と一致します。オプティマイザ状態を保持するには `reset_optimizer_at_freeze: false` を設定してください。

MFCC と FBank は同じ総エポック数を使用し、AFSC パラメータを最適化しません:

```bash
python -m Model.train --config Model/config.yaml --feature fbank --train-csv /data/train/train.csv --output runs/fbank --device cuda
python -m Model.train --config Model/config.yaml --feature mfcc --train-csv /data/train/train.csv --output runs/mfcc --device cuda
```

出力: `config.yaml`、`speakers.json`、エポックごとの `train.jsonl`、`epoch_001.pt` など、`last.pt`、およびエポックごとの AFSC 周波数点 JSON。抽出に必要なすべての重みと AFSC バッファは各チェックポイントに含まれます。完全なチェックポイントには分類器/オプティマイザ状態、解決済みマニフェスト、RNG 状態も含まれるため、ローカルの学習パスが含まれます。チェックポイントを公開する前にこれらのパスを確認してください。

**次のエポック**から再開します:

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
```

同一のレシピ、特徴オーバーライド（ある場合）、マニフェストを使用してください。`--stop-after-epoch 2` は、完全な学習率スケジュールを変更せずにエポック境界でクリーンに停止します。エポック途中からの再開は実装されていません。新しい結果があるディレクトリで古いチェックポイントを再開する場合は、新しい出力ディレクトリを使用してください。CPU でのエポック境界再開は、小さなレシピで厳密な等価性がテストされています。ハードウェア/CUDA 環境間での厳密な等価性は保証されません。

## 抽出と評価

```bash
python -m Inference.extract --checkpoint runs/afsc/last.pt --csv /data/eval/eval.csv --output runs/afsc/eval.npz --device cuda
python -m Evaluation.score --enrol runs/afsc/eval.npz --trials /data/eval/trials --output runs/afsc/scores
```

登録用とテスト用のアーカイブが別々の場合は、スコアリングに `--test runs/afsc/test.npz` を追加します。抽出はデフォルトで発話全体を使用します。0.1 秒未満のクリップは、エンコーダ互換性のためにゼロ埋めされます。任意の `--chunk-seconds 30` は 30 秒の非オーバーラップチャンクを使用します。2 秒以上の残り末尾がある場合は、末尾揃えのチャンクを追加します。生のチャンク埋め込みは平均化され、その後一度だけ L2 正規化されます。この任意の長音声パスはターゲットドメイン適応を行いません。

埋め込みは NPZ 配列 `ids`（文字列）と `embeddings`（N x D）として保存され、pickle なしで読み込まれます。スコアリングは `scores.csv` と `metrics.json` を書き出します。EER は**パーセンテージ**で表され、MinDCF は正規化比率です。デフォルトの DCF パラメータは `p_target=0.01, c_miss=1, c_fa=1` です。メトリクスは丸められていないスコアから計算され、等しいスコアは単一の動作点にグループ化されます。

二波形推論:

```bash
python -m Inference.verify --checkpoint runs/afsc/last.pt --enrol /data/a.wav --test /data/b.wav --device cpu
```

これはコサインスコアを返します。`--threshold VALUE` は、適切な開発試行で閾値を校正した後にのみ追加してください。普遍的なデフォルト閾値はありません。

## AFSC の詳細と元の作業スクリプトとの違い

- 時間前処理は、提供されたスペクトル抽出器を保持します: 25 ms フレーム、10 ms シフト、フレーム DC 除去、0.97 プリエンファシス、Povey 窓、512 点 FFT、257 パワービン。パワーは FFT サイズで除算されません。AFSC には対数圧縮とチャネルごとの時間平均減算があり、DCT や離散フィルタ和正規化はありません。
- `Features/afsc.py` は、**レガシーな 82 エントリ参照シーケンスとその正規化正ギャップパラメータ化**を保持します。80 個の学習可能なギャップパラメータがあり、累積正規化により上限端点が繰り返されます。初期フォワードパスの点は、生の参照リストとは異なります。その範囲は厳密に 20–7600 Hz ではなく、おおよそ 20.039–7614.844 Hz です。これは提供された手法を黙って変更するのではなく、文書化されています。Float32 の丸めにより、ほぼ一致する点が生成されることがあります。最小ギャップは課されません。
- 区分フィルタ分母は、幅の二乗に `1e-12` を加えたものです。分岐は連続的に接続する必要はありません。FFT サポートがないフィルタは、一定のフロア付き対数エネルギーと、平均正規化チャネルがゼロになります。
- MFCC/FBank は torchaudio 2.5.1 の Kaldi デフォルトに加え、明示的に選択された 80 チャネル、dither=0、平均減算を使用します。MFCC は 80 Mel ビンと 80 ケプストラムを使用し、デフォルトのケプストラルリフタを含みます。デフォルトの周波数範囲は 20–8000 Hz です。したがって、フロントエンドはフィルタ形状だけが異なるわけではありません。
- 公開デフォルトは、原稿の **2+8 エポックと一定マージン 0.3** に従います。提供された履歴ログは **2+10 エポックと 0 から 0.3 へのマージンランプ**で実行されました。`Model/legacy_schedule.yaml` はそのスケジュールを公開し、10 エポックの LR 減衰ホライズンを含みます。これは過去のスコアを復元することを約束するものではありません。ソースデータリストやスコアリングチェックポイントは特定されていません。
- 新しいデータ変換器は 1 発話につき 1 行を使用します。古い固定チャンク CSV はパスを繰り返すことがあり、その学習リーダーはセグメントオフセットを無視していました。ここでは明示的なセグメントオフセットが反映されます。これにより、古い CSV ファイルと比較してエポックごとのサンプリングが変わる可能性があります。
- 学習/分類器/チェックポイントの配線は、スタンドアロン PyTorch として書き直されています。レガシーな `embedding_model.ckpt` ファイルは直接読み込めません。このリリースで学習してください。ECAPA エンコーダのソースは保持されています。固定特徴の時間前処理は、提供されたプロセッサから保持されています。
- EER/MinDCF は同点対応閾値を使用し、端点動作点を含みます。これは、提供された逐次スコア実装よりも等しいスコアに対して堅牢で、同点時に異なる場合があります。報告された数値との一致を強制するためにスコアリングを変更することはありません。

## テスト

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

スモークテストは、一時的な合成音声を作成し、小さなモデルを学習し、凍結と厳密なエポック再開をテストし、埋め込みを抽出/比較し、試行をスコアリングし、フィルタを出力し、MFCC/FBank を実行します。CN-Celeb で学習したり、論文レベルの数値再現性を確立したりするものではありません。このリリースで実行された検証については `TESTING.md` を参照してください。

## 帰属

`THIRD_PARTY_NOTICES.md` と `LICENSE` を参照してください。このリリースは、ユーザー提供の 3D-Speaker/SpeechBrain 由来の ECAPA および前処理コードに基づいており、帰属表示は保持されています。提案された特徴手法を使用する場合は、AFSC 原稿を引用してください。研究用音声は、その元の提供元から入手する必要があります。
