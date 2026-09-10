# AFSC + ECAPA-TDNN: training and speaker verification

A standalone implementation of the first part of **Adaptive Frequency Spectral Coefficients for Deep Speaker Verification: Training and Cross-Domain Adaptation**.

This release provides the AFSC front end and an ECAPA-TDNN training, embedding extraction and evaluation pipeline. MFCC and FBank can be selected as baselines with the same encoder. Res2Net, X-Vector and target-domain adaptation are outside this release's scope. The repository does not redistribute research datasets or pretrained research checkpoints, and does not claim to reproduce an unidentified historical checkpoint or all numerical results in the manuscript.

## Main directories

| Directory | Contents |
|---|---|
| `Dataset` | Audio/CSV loading, wav.scp + utt2spk conversion, synthetic demo generator |
| `Features` | Power spectrum, MFCC/FBank, trainable AFSC and filter export |
| `Model` | ECAPA, cosine classifier, angular-margin loss, two-stage training and YAML recipes |
| `Evaluation` | Trial scoring, EER/MinDCF, unit and integration tests |
| `Inference` | Checkpoint loading, utterance embeddings and two-waveform comparison |

All commands below run **from this repository's root directory**. On Windows the same `python -m ...` commands work; use `num_workers: 0` if multiprocessing is inconvenient. No installation of `speakerlab`, Kaldi binaries, or `kaldiio` is needed. Torchaudio's Kaldi-compatible Python functions are used for the fixed front ends.

## Installation

Use Python 3.10–3.12. The included validation was run with Python 3.12 on CPU.

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\activate` on Windows cmd. Install matching PyTorch and torchaudio builds. For CPU:

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install a compatible matching PyTorch/torchaudio 2.5.1 build for the machine, then install `requirements.txt`. The reference model is large; use the demo recipe for a quick CPU check.

## Quick end-to-end example

Generate artificial tones and small train/evaluation manifests:

```bash
python -m Dataset.demo --output demo_data
python -m Model.train --config Model/demo.yaml --train-csv demo_data/train.csv --output runs/demo --device cpu
python -m Inference.extract --checkpoint runs/demo/last.pt --csv demo_data/eval.csv --output runs/demo/embeddings.npz --device cpu
python -m Evaluation.score --enrol runs/demo/embeddings.npz --trials demo_data/trials.txt --output runs/demo/scores
python -m Features.export --checkpoint runs/demo/last.pt --output runs/demo/filters --plot
```

**The demo uses synthetic tones and a reduced ECAPA. Its scores are software checks, not speaker-verification research results.** Its train and evaluation synthetic identities are disjoint. No real person's speech is included.

## Prepare research data

Obtain the datasets from their providers. Supply separate training and evaluation manifests using the intended dataset protocol. No automatic random train/test split or randomly generated research trials is performed. Avoid training/evaluation speaker leakage and choose any hyperparameters or operating threshold on separate development data.

`wav.scp` format (one utterance per line):

```text
utt001 /path/to/audio001.wav
utt002 /path/to/audio002.flac
```

Paths with spaces are supported. Relative paths are resolved relative to `wav.scp`. Shell pipelines such as `sox ... |` are not supported. `utt2spk` contains:

```text
utt001 speaker001
utt002 speaker002
```

Create a training CSV:

```bash
python -m Dataset.prepare --wav-scp /data/train/wav.scp --utt2spk /data/train/utt2spk --output /data/train/train.csv
```

Create an evaluation CSV (speaker labels are optional):

```bash
python -m Dataset.prepare --wav-scp /data/eval/wav.scp --output /data/eval/eval.csv
```

CSV columns are `ID,path,spk,dur`; only `ID,path` are required for extraction and `spk` is additionally required for training. Relative CSV paths resolve beside the CSV. Audio must already be **16 kHz**; channel zero is used for multichannel audio. Each training row yields one random 3-second crop, with short recordings right-zero-padded. The default converter emits one row per utterance. Optional `start,stop` CSV fields, measured in seconds, are honored before random cropping. The converter does not generate these fields.

Supply evaluation trials as:

```text
utt001 utt002 0
utt001 utt003 1
```

Labels can be `0`/`1` or `nontarget`/`target`. All IDs must occur in the corresponding embedding archives. One ID denotes one utterance embedding; multi-utterance enrollment aggregation, VAD and score normalization are not implicit. If your dataset protocol requires enrollment aggregation or preprocessing, implement and document that protocol before evaluation.

## Train AFSC + ECAPA

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --device cuda
```

The default encoder has channels `[1024,1024,1024,1024,3072]`, 80 input feature channels and 192 output dimensions. Training uses a normalized cosine classifier and angular-margin loss (`scale=32`, `margin=0.3`). AFSC gap parameters train jointly for 2 epochs with a 30x learning-rate multiplier and zero weight decay, then remain frozen for 8 epochs. SGD momentum is reset at the stage transition by default, matching the supplied script's optimizer rebuild. Set `reset_optimizer_at_freeze: false` to retain the optimizer state instead.

MFCC and FBank use the same total epoch count and do not optimize AFSC parameters:

```bash
python -m Model.train --config Model/config.yaml --feature fbank --train-csv /data/train/train.csv --output runs/fbank --device cuda
python -m Model.train --config Model/config.yaml --feature mfcc --train-csv /data/train/train.csv --output runs/mfcc --device cuda
```

Outputs: `config.yaml`, `speakers.json`, per-epoch `train.jsonl`, `epoch_001.pt` etc., `last.pt`, and per-epoch AFSC frequency-point JSON. All weights and AFSC buffers needed for extraction are in each checkpoint. Full checkpoints also contain classifier/optimizer state, the resolved manifest and RNG states; they therefore contain local training paths. Review those paths before publishing a checkpoint.

Resume at the **next epoch**:

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
```

Use the identical recipe, feature override (if any), and manifest. `--stop-after-epoch 2` cleanly stops at an epoch boundary without changing the full learning-rate schedule. Mid-epoch resume is not implemented. Resuming an older checkpoint in a directory with newer results should use a new output directory. CPU epoch-boundary resume is tested for exact equality on the tiny recipe; exact equality across hardware/CUDA environments is not guaranteed.

## Extract and evaluate

```bash
python -m Inference.extract --checkpoint runs/afsc/last.pt --csv /data/eval/eval.csv --output runs/afsc/eval.npz --device cuda
python -m Evaluation.score --enrol runs/afsc/eval.npz --trials /data/eval/trials --output runs/afsc/scores
```

For separate enrollment/test archives, add `--test runs/afsc/test.npz` to scoring. Extraction defaults to the entire utterance. Clips shorter than 0.1 seconds are zero-padded for encoder compatibility. Optional `--chunk-seconds 30` uses 30-second nonoverlapping chunks; a remaining tail of at least 2 seconds adds an end-aligned chunk. Raw chunk embeddings are averaged, then L2-normalized once. This optional long-audio path does not perform target-domain adaptation.

Embeddings are stored as NPZ arrays `ids` (strings) and `embeddings` (N x D), loaded without pickle. Scoring writes `scores.csv` and `metrics.json`. EER is expressed as a **percentage**, MinDCF as a normalized ratio. Default DCF parameters are `p_target=0.01, c_miss=1, c_fa=1`. Metrics are computed from unrounded scores and equal scores are grouped into a single operating point.

Two-waveform inference:

```bash
python -m Inference.verify --checkpoint runs/afsc/last.pt --enrol /data/a.wav --test /data/b.wav --device cpu
```

This returns a cosine score. Add `--threshold VALUE` only after calibrating a threshold on appropriate development trials; there is no universal default threshold.

## AFSC details and differences from the original working scripts

- Temporal preprocessing preserves the supplied spectrum extractor: 25 ms frames, 10 ms shift, frame DC removal, 0.97 pre-emphasis, Povey window, 512-point FFT and 257 power bins. Power is not divided by FFT size. AFSC has log compression and per-channel temporal mean subtraction, no DCT or discrete filter-sum normalization.
- `Features/afsc.py` preserves the **legacy 82-entry reference sequence and its normalized positive-gap parameterization**. It has 80 trainable gap parameters; cumulative normalization produces a repeated upper endpoint. The initial forward-pass points differ from the raw reference list. Its bounds are approximately 20.039–7614.844 Hz, rather than exactly 20–7600 Hz. This is documented rather than silently changing the supplied method. Float32 rounding can produce nearly coincident points. No minimum gap is imposed.
- Piecewise filter denominators are squared widths plus `1e-12`. Branches need not join continuously. Filters with no FFT support give constant floored log energy and zero mean-normalized channels.
- MFCC/FBank use torchaudio 2.5.1 Kaldi defaults plus explicitly selected 80 channels, dither=0 and mean subtraction. MFCC uses 80 Mel bins and 80 cepstra, including the default cepstral lifter. Their default frequency bounds are 20–8000 Hz. Thus the front ends do not differ in filter shape alone.
- The public default follows the manuscript's **2+8 epochs and constant margin 0.3**. The supplied historical log ran **2+10 epochs with a margin ramp from 0 to 0.3**. `Model/legacy_schedule.yaml` exposes that schedule, including an LR decay horizon of 10 epochs; it is not a promise of recovering a historical score. No source data list or scoring checkpoint was identified.
- The new data converter uses one row per utterance. The old fixed-chunk CSV could repeat paths while its training reader ignored segment offsets. Here explicit segment offsets are honored. This can change per-epoch sampling relative to old CSV files.
- Training/classifier/checkpoint wiring is rewritten as standalone PyTorch. Legacy `embedding_model.ckpt` files cannot be loaded directly; train with this release. The ECAPA encoder source is retained. Fixed-feature temporal preprocessing is retained from the supplied processor.
- EER/MinDCF use tie-aware thresholds and include endpoint operating points. This is more robust for equal scores than the supplied sequential-score implementation and can differ on ties. No change is made to scoring to force agreement with a reported number.

## Tests

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

The smoke test creates temporary synthetic audio, trains a small model, tests freezing and exact epoch resume, extracts/compares embeddings, scores trials, exports filters, and exercises MFCC/FBank. It does not train on CN-Celeb or establish paper-level numerical reproducibility. See `TESTING.md` for the validation performed for this release.

## Attribution

See `THIRD_PARTY_NOTICES.md` and `LICENSE`. This release builds on user-supplied 3D-Speaker/SpeechBrain-derived ECAPA and preprocessing code; attribution is retained. Cite the AFSC manuscript when using the proposed feature method. Research audio must be obtained from its original providers.
