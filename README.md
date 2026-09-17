# AFSC for Deep Speaker Verification and Cross Domain Adaptation

**English** | [中文](README_中文.md) | [日本語](README_日本語.md)

This project covers AFSC features, ECAPA training, embedding extraction, speaker verification, and a target-domain adaptation demo. MFCC and FBank baselines are retained. Official Res2Net and X-Vector code is included. The code can be retrained from scratch and also supports loading the CN-Celeb2 pretrained models provided with the paper. The cross-domain adaptation part currently provides a lightweight demo that verifies the residual adapter, target-domain embedding standardization, and fixed score fusion pipeline.

The project supports two usage modes:

- **Train your own model**: start from `Speech_example/` or your own dataset and train AFSC + ECAPA with `Model/train.py`.
- **Load pretrained models**: convert the paper's `embedding_model.ckpt` + `classifier.ckpt` into this project's checkpoint format using `tools/convert_legacy.py`, then extract, score, and compare directly.

Both modes share the same `Inference/`, `Evaluation/`, and `Features/` scripts. The cross-domain adaptation part additionally uses the `Adaptation/` directory.

## Main directories

| Folder | Main files | Purpose |
|---|---|---|
| Dataset | data.py, prepare.py, demo.py | Read audio, prepare training manifests, generate demo data from speaker folders |
| Features | spectrum.py, afsc.py, frontend.py, export.py | Power spectrum and three acoustic features, export learned filters |
| Model | ecapa_tdnn.py, system.py, train.py | ECAPA, classification loss, two-stage training and resume |
| Evaluation | metrics.py, score.py, test_pipeline.py, smoke_test.py | Scoring, EER/MinDCF, verification code |
| Inference | extract.py, verify.py | Extract embeddings, compare two audio files |
| Adaptation | adapter.py, losses.py, chunking.py, standardize.py, trials.py, sampler.py, dataset.py, embed.py, train.py, demo.py | Target-domain adaptation: residual adapter, embedding standardization, fixed fusion, AFSC gap update |

In addition:

| Folder | Contents |
|---|---|
| `Speech_example/` | Example audio, one speaker per subfolder, 15 utterances per folder |
| `PreTrained/` | Pretrained checkpoints provided with the paper (`embedding_model.ckpt` + `classifier.ckpt`) and the converted `cn_celeb2_afsc_ecapa.pt` |
| `tools/` | Helper scripts, including `convert_legacy.py` and `inspect_ckpt.py` |

Training code lives in Model, so no separate Training directory is added. All commands are run from the project root.

## Installation

Use Python 3.10–3.12. First install the matching PyTorch and torchaudio 2.5.1, then install requirements.txt. CPU example:

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

GPU users can install the corresponding CUDA build; `requirements.txt` does not need to change.

## Run the demo first

Generate manifests from `Speech_example/`:

```bash
python -m Dataset.demo
```

Expected output:

```
Train:  30 utterances from 3 speakers
Eval:   15 utterances from 3 speakers
Trials: 105 (30 target, 75 nontarget)
Manifests written to .../demo_data
```

Generated files in `demo_data/`:

| File | Contents |
|---|---|
| `train.csv` | Training manifest, each line `ID,path,spk` |
| `eval.csv` | Evaluation manifest, each line `ID,path,spk` |
| `trials.txt` | Evaluation trials, each line `enrol_id test_id label` |

By default each speaker keeps 5 utterances for evaluation and the rest for training. To change:

```bash
python -m Dataset.demo --eval-per-speaker 3
```

## Full pipeline for training your own model

In PyCharm, right-click and Run the following files in order; or run them from the command line with `python -m ...`.

### 1. Training

```bash
python -m Model.train --config Model/demo.yaml --train-csv demo_data/train.csv --output runs/demo --device cpu
```

`Model/demo.yaml` is a CPU-friendly small model; `Model/config.yaml` is the paper configuration (2+8 epochs, `channels=[1024,1024,1024,1024,3072]`, 192-dim embedding) used for formal experiments.

Training output is in `runs/demo/`:

| File | Contents |
|---|---|
| `config.yaml` | Full configuration used for this run |
| `speakers.json` | Speaker → index mapping |
| `train.jsonl` | Per-epoch loss / accuracy / lr / margin |
| `epoch_XXX.pt` | Full checkpoint for each epoch |
| `last.pt` | Copy of the last epoch |
| `frequency_points_XXX.json` | AFSC frequency points for each epoch |

### 2. Extract evaluation embeddings

```bash
python -m Inference.extract --checkpoint runs/demo/last.pt --csv demo_data/eval.csv --output runs/demo/embeddings.npz --device cpu
```

Output:

- `runs/demo/embeddings.npz`: contains `ids` (string array) and `embeddings` (N × D)
- `runs/demo/embeddings.json`: metadata recording checkpoint, csv, feature, utterances, chunk_seconds

### 3. Scoring

```bash
python -m Evaluation.score --enrol runs/demo/embeddings.npz --trials demo_data/trials.txt --output runs/demo/scores
```

Output:

- `runs/demo/scores/scores.csv`: each trial's enrol_id, test_id, label, score
- `runs/demo/scores/metrics.json`: EER, MinDCF, trial counts

EER is reported as a percentage, MinDCF uses the normalized value, and the default target prior is `0.01`.

### 4. Export filters (optional)

```bash
python -m Features.export --checkpoint runs/demo/last.pt --output runs/demo/filters --plot
```

Output:

- `runs/demo/filters/afsc_filters.npz`: contains `bin_points` (82,), `hz` (82,), `filters` (80, 257)
- `runs/demo/filters/afsc_filters.png` (if `--plot` is given)

### 5. Compare two audio files (optional)

```bash
python -m Inference.verify --checkpoint runs/demo/last.pt --enrol Speech_example/speech_demo1/speech-01-001.flac --test Speech_example/speech_demo1/speech-01-002.flac
```

If `--enrol` / `--test` are omitted, the first two audio files under `Speech_example/speech_demo1/` are used. `--threshold` must be calibrated on a development set; there is no universal default.

## Cross-domain adaptation demo

The `Adaptation/` directory compresses the target-domain adaptation procedure from Section 3.5 of the paper onto the 3-speaker `demo_data/` set to verify the pipeline. The procedure includes:

- Freeze the pretrained speaker encoder;
- Append a residual adapter after the 192-dim embedding: `192 → 384 → 192`, ReLU, Dropout 0.40, zero-initialized fc2, `alpha` initialized to 0.1;
- Training loss: supervised contrastive loss + 0.05 × identity MSE;
- P=8 / K=4 PK sampling, 100 batches per epoch;
- Target-domain embedding component-wise standardization followed by L2 normalization;
- Fixed 0.5/0.5 score fusion;
- AFSC features additionally support target-domain gap updates (default 2 epochs in the demo).

It depends on the already-completed Part 1 artifacts:

```text
runs/demo/last.pt
demo_data/train.csv
demo_data/eval.csv
```

Run the demo:

```bash
python -m Adaptation.demo --device cpu
```

To train only the adapter without triggering AFSC gap updates:

```bash
python -m Adaptation.train --device cpu --gap-epochs 0
```

Full pipeline (AFSC runs 2 epochs of gap update by default):

```bash
python -m Adaptation.train --device cpu
```

Output is in `runs/demo_adapt/` (or `runs/demo_train/`):

| File | Contents |
|---|---|
| `summary.json` | EER and MinDCF for baseline / standardized / adapter / fusion |
| `history.json` | Per-epoch loss / supcon / identity / alpha of the adapter |
| `gap_history.json` | Only generated when `--gap-epochs > 0`, records the AFSC gap update process |

Meaning of the four blocks in `summary.json`:

| Field | Meaning | Paper column |
|---|---|---|
| `baseline` | Unadapted cosine scores directly from the pretrained embedding | Before |
| `standardized` | Target-domain component-wise standardization + L2 scores | Standardized |
| `adapter` | Scores after training the residual adapter | Adapter |
| `fusion` | `0.5 × adapter + 0.5 × standardized` | Fusion |

The demo uses a small model (`Model/demo.yaml`) and does not reproduce the paper's numbers. Real SITW cross-domain experiments require `Adaptation/run.py` (not yet packaged for public release) and the full SITW Dev / Eval protocol.

## Loading the pretrained model

The paper's authors provide AFSC + ECAPA weights trained on CN-Celeb2:

```
PreTrained/embedding_model.ckpt    # AFSC + ECAPA weights
PreTrained/classifier.ckpt         # 2793 × 192 classifier head
```

These are in the legacy format and must first be converted to this project's `format_version: 1` with `tools/convert_legacy.py`:

```bash
python tools/convert_legacy.py
```

The conversion script will:

1. Rename the `afsc.*` prefix to `frontend.*`;
2. Rename the `ecapa.*` prefix to `encoder.*`;
3. Verify that all keys match `SpeakerSystem`;
4. Write `PreTrained/cn_celeb2_afsc_ecapa.pt` and `PreTrained/cn_celeb2_afsc_ecapa.json`.

If you see `All keys matched exactly.`, the conversion succeeded completely.

### Extract and score with the pretrained model

To switch, open `Inference/extract.py`, `Evaluation/score.py`, `Features/export.py`, `Inference/verify.py` and comment out section (A) and uncomment section (B) in `default_checkpoint()` (or `default_paths()`).

Then you can use the same commands as the "train your own model" workflow:

```bash
python -m Inference.extract --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --csv demo_data/eval.csv --output runs/pretrained_eval.npz --device cpu
python -m Evaluation.score --enrol runs/pretrained_eval.npz --trials demo_data/trials.txt --output runs/pretrained_scores
python -m Features.export --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --output runs/pretrained_filters --plot
```

### Notes on the pretrained model

- The 3 speakers in `Speech_example/` are **not** among the 2793 speakers in CN-Celeb2. The pretrained model has never seen them, so the EER on this demo may be high; this is expected.
- Meaningful evaluation requires the official CN-Celeb2 or SITW protocols and trial lists.
- The pretrained model was trained with the AFSC front end, so it can only be used with `feature: afsc`.
- The speaker mapping stored in the checkpoint is a placeholder (`speaker_00000`…`speaker_02792`) and only affects mapping classification outputs back to real speaker names.

## Using real data

Prepare 16 kHz audio plus wav.scp and utt2spk for the training set. wav.scp has one line per utterance (`audio_id path`), utt2spk has one line per utterance (`audio_id speaker_id`). Audio files are not copied into the repository.

```bash
python -m Dataset.prepare --wav-scp /data/train/wav.scp --utt2spk /data/train/utt2spk --output /data/train/train.csv
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --device cuda
```

Without a GPU you can use `--device cpu`, but training the full model will be very slow. On Windows, if you encounter multiprocessing issues, set `num_workers` to 0 in the YAML.

Extracting a test set does not require speaker labels:

```bash
python -m Dataset.prepare --wav-scp /data/eval/wav.scp --output /data/eval/eval.csv
python -m Inference.extract --checkpoint runs/afsc/last.pt --csv /data/eval/eval.csv --output runs/afsc/eval.npz --device cuda
python -m Evaluation.score --enrol runs/afsc/eval.npz --trials /data/eval/trials --output runs/afsc/scores
```

Each line of trials is `enrol_audio_id test_audio_id label`, with 1 for the same speaker and 0 for different speakers. Use splits and trials that match your actual dataset protocol. The code will not guess the CN-Celeb enrollment aggregation method and will not generate a random research test set to substitute for the official protocol.

Add `--feature fbank` or `--feature mfcc` to the training command to switch baselines; also change the output directory. Extraction reads the feature type from the checkpoint automatically, so no need to specify it again.

The cross-domain adaptation part also accepts the same CSV format. `Adaptation.train` and `Adaptation.demo` read manifests through `--train-csv` / `--eval-csv`, take a pretrained model through `--checkpoint`, and control long-audio handling with `--chunk-seconds` (`None` for whole-utterance processing, 30.0 for 30-second chunks).

## Differences from the original code

- `Model/config.yaml`: paper configuration, 2+8 epochs, margin fixed at 0.3, `channels=[1024,1024,1024,1024,3072]`, embedding dimension 192.
- `Model/legacy_schedule.yaml`: retains the training schedule options corresponding to the log, 2+10 epochs, margin growing from 0 to 0.3, learning rate reaching its minimum at the end of epoch 10. It can only express the training schedule, not prove correspondence to a specific paper result.
- `Model/demo.yaml`: reduced model, CPU quick-validation only.
- AFSC retains the original code's initialization reference array, positive-gap normalization, and filter formula. Its exact boundaries are approximately 20.039–7614.844 Hz, and the first actual output points are also affected by normalization. It has not been quietly replaced with a new "exact 20–7600 Hz initialization".
- New CSVs default to one row per utterance and randomly crop 3 seconds during training. If a user supplies start/stop, the segment is read first and then cropped. The original behavior of "multiple CSV rows with segments but training ignoring the interval" is not carried over.
- The new trainer saves full state for resuming and uses the new checkpoint format. The legacy `embedding_model.ckpt` cannot be loaded directly and must first be converted with `tools/convert_legacy.py`.
- Scoring explicitly handles tied scores and ROC endpoints, so it may differ slightly from the legacy metric function when scores are tied.
- The `Adaptation/` residual adapter follows Section 3.5.2 of the paper; the identity loss is computed against a detached base embedding so that AFSC gap updates do not pull the filter toward identity.
- During AFSC gap updates, the encoder stays in `eval()` and only `raw_gaps` and the adapter are updated; Res2Net and X-Vector do not perform additional gap updates in the paper.

## Resuming training and inspecting results

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
python -m Features.export --checkpoint runs/afsc/last.pt --output runs/afsc/filters --plot
```

Training output includes per-epoch logs, configuration, speaker indices, full checkpoints, and frequency points. Resuming goes to the next epoch; mid-epoch resume is not supported. Full checkpoints contain data manifest paths, which should be inspected before releasing models publicly.

## Checks

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

See TESTING.md for the current test scope. Passing tests means the software pipeline runs; it does not replace training and experimental validation on real datasets.

## Attribution

See `THIRD_PARTY_NOTICES.md` and `LICENSE`. This release is built on the user-provided 3D-Speaker / SpeechBrain-derived ECAPA and preprocessing code, with attribution retained. When using the proposed feature method, please cite the AFSC paper. Research audio must be obtained from the original providers.

`tools/convert_legacy.py` is original code in this project, used to convert 3D-Speaker-style AFSC + ECAPA checkpoints into this project's format. The converted checkpoint contents remain subject to the original licenses. Confirm you have redistribution rights before using pretrained weights.

The residual adapter, supervised contrastive loss, target-domain embedding standardization, and fixed score fusion in `Adaptation/` follow Section 3.5 of the paper and the author-provided experimental notes. The full SITW experiment code on Colab depends on `speakerlab` and Google Drive and is not included in this public repository.

### Demo dataset

The demo audio in `Speech_example/` comes from the CN-Celeb dataset (OpenSLR SLR82), used to demonstrate the pipeline; the full dataset is not redistributed.

- Dataset homepage: https://openslr.org/82/
- License: Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- Source: released by the Center for Speech and Language Technologies (CSLT), Tsinghua University

CN-Celeb citation:

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

When using the demo audio or code in this repository, please comply with CC BY-SA 4.0 and retain the attribution above. This repository does not distribute the full CN-Celeb dataset, only a small number of demo audio files. For the full dataset, obtain it from https://openslr.org/82/.
