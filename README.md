# AFSC + ECAPA Part 1 Public Code

**English** | [中文](README.zh.md) | [日本語](README.ja.md)


This project covers the first part of the paper: AFSC features, ECAPA training, embedding extraction, and speaker verification, with MFCC and FBank baselines. Cross-domain adaptation is not included, and neither Res2Net nor X-Vector is included. The code can be retrained and can also load the CN-Celeb2 pretrained model provided with the paper.

The project supports two workflows:

- **Train your own model**: start from `Speech_example/` or your own dataset and train AFSC + ECAPA with `Model/train.py`;
- **Load the pretrained model**: convert the paper-provided `embedding_model.ckpt` + `classifier.ckpt` into this project's checkpoint format with `tools/convert_legacy.py`, then extract, score, and compare directly.

Both workflows share the same `Inference/`, `Evaluation/`, and `Features/` scripts.

## Main directories

| Directory | Main files | Purpose |
|---|---|---|
| Dataset | data.py, prepare.py, demo.py | Audio loading, training manifest preparation, demo data generation from speaker folders |
| Features | spectrum.py, afsc.py, frontend.py, export.py | Power spectrum and three acoustic features, learned filter export |
| Model | ecapa_tdnn.py, system.py, train.py | ECAPA, classification loss, two-stage training and resume |
| Evaluation | metrics.py, score.py, test_pipeline.py, smoke_test.py | Scoring, EER/MinDCF, validation code |
| Inference | extract.py, verify.py | Embedding extraction, two-waveform comparison |

Additional directories:

| Directory | Contents |
|---|---|
| `Speech_example/` | Example audio. Each subfolder is one speaker, with 15 utterances per folder |
| `PreTrained/` | Paper-provided pretrained checkpoints (`embedding_model.ckpt` + `classifier.ckpt`) and the converted `cn_celeb2_afsc_ecapa.pt` |
| `tools/` | Utility scripts, including `convert_legacy.py` and `inspect_ckpt.py` |

Training code lives under Model, so no sixth Training directory is added. All commands are run from the project root.

## Installation

Use Python 3.10–3.12. Install matching PyTorch and torchaudio 2.5.1 first, then install requirements.txt. CPU example:

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

GPU users can install the corresponding CUDA build; `requirements.txt` does not need to change.


## First run the demo

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

Generated files under `demo_data/`:

| File | Contents |
|---|---|
| `train.csv` | Training manifest, each line `ID,path,spk` |
| `eval.csv` | Evaluation manifest, each line `ID,path,spk` |
| `trials.txt` | Evaluation trials, each line `enrol_id test_id label` |

By default, 5 utterances per speaker are held out for evaluation and the rest are used for training. You can change this:

```bash
python -m Dataset.demo --eval-per-speaker 3
```

## Full self-training workflow

Right-click Run the following files in order in PyCharm, or run them from the command line with `python -m ...`.

### 1. Train

```bash
python -m Model.train --config Model/demo.yaml --train-csv demo_data/train.csv --output runs/demo --device cpu
```

`Model/demo.yaml` is a small CPU-friendly model; `Model/config.yaml` is the paper configuration (2+8 epochs, `channels=[1024,1024,1024,1024,3072]`, 192-d embeddings) and should be used for formal experiments.

Training outputs under `runs/demo/`:

| File | Contents |
|---|---|
| `config.yaml` | Full configuration used for this run |
| `speakers.json` | Speaker → index mapping |
| `train.jsonl` | Per-epoch loss / accuracy / lr / margin |
| `epoch_XXX.pt` | Full checkpoint for each epoch |
| `last.pt` | Copy of the final epoch |
| `frequency_points_XXX.json` | AFSC frequency points for each epoch |

### 2. Extract evaluation embeddings

```bash
python -m Inference.extract --checkpoint runs/demo/last.pt --csv demo_data/eval.csv --output runs/demo/embeddings.npz --device cpu
```

Outputs:

- `runs/demo/embeddings.npz`: contains `ids` (string array) and `embeddings` (N × D)
- `runs/demo/embeddings.json`: metadata recording checkpoint, csv, feature, utterances, chunk_seconds

### 3. Score

```bash
python -m Evaluation.score --enrol runs/demo/embeddings.npz --trials demo_data/trials.txt --output runs/demo/scores
```

Outputs:

- `runs/demo/scores/scores.csv`: enrol_id, test_id, label, score for every trial
- `runs/demo/scores/metrics.json`: EER, MinDCF, trial counts

EER is expressed as a percentage, MinDCF as a normalized ratio, and the default target prior is `0.01`.

### 4. Export filters (optional)

```bash
python -m Features.export --checkpoint runs/demo/last.pt --output runs/demo/filters --plot
```

Outputs:

- `runs/demo/filters/afsc_filters.npz`: contains `bin_points` (82,), `hz` (82,), `filters` (80, 257)
- `runs/demo/filters/afsc_filters.png` (when `--plot` is given)

### 5. Compare two waveforms (optional)

```bash
python -m Inference.verify --checkpoint runs/demo/last.pt --enrol Speech_example/speech_demo1/speech-01-001.flac --test Speech_example/speech_demo1/speech-01-002.flac
```

If `--enrol` / `--test` are omitted, the first two files under `Speech_example/speech_demo1/` are used by default. `--threshold` must be calibrated on a development set; there is no universal default.

## Loading the pretrained model

The paper authors provide CN-Celeb2-trained AFSC + ECAPA weights:

```
PreTrained/embedding_model.ckpt    # AFSC + ECAPA weights
PreTrained/classifier.ckpt         # 2793 × 192 classification head
```

These are legacy format and must be converted to this project's `format_version: 1` with `tools/convert_legacy.py`:

```bash
python tools/convert_legacy.py
```

The conversion script will:

1. rename the `afsc.*` prefix to `frontend.*`;
2. rename the `ecapa.*` prefix to `encoder.*`;
3. validate that every key matches `SpeakerSystem`;
4. write `PreTrained/cn_celeb2_afsc_ecapa.pt` and `PreTrained/cn_celeb2_afsc_ecapa.json`.

If you see `All keys matched exactly.`, the conversion succeeded completely.

### Extract and score with the pretrained model

To switch: open `Inference/extract.py`, `Evaluation/score.py`, `Features/export.py`, and `Inference/verify.py`, comment out block (A) in `default_checkpoint()` (or `default_paths()`), and uncomment block (B).

Then use the same commands as for self-training:

```bash
python -m Inference.extract --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --csv demo_data/eval.csv --output runs/pretrained_eval.npz --device cpu
python -m Evaluation.score --enrol runs/pretrained_eval.npz --trials demo_data/trials.txt --output runs/pretrained_scores
python -m Features.export --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --output runs/pretrained_filters --plot
```

### Notes on the pretrained model

- The 3 speakers in `Speech_example/` are **not** among the 2793 CN-Celeb2 speakers. The pretrained model has never seen them, so EER on this demo may be high; this is expected.
- Meaningful evaluation requires the official CN-Celeb2 or SITW evaluation protocol and trial lists.
- The pretrained model was trained with the AFSC front end, so it can only be used with `feature: afsc`.
- The speaker mapping saved in the checkpoint is a placeholder (`speaker_00000`…`speaker_02792`); it only affects mapping classification outputs back to real speaker names.

## Switching to real data

Prepare 16 kHz audio plus the training wav.scp and utt2spk. Each line of wav.scp is "audio ID path"; each line of utt2spk is "audio ID speaker ID". Audio files are not copied into the repository.

```bash
python -m Dataset.prepare --wav-scp /data/train/wav.scp --utt2spk /data/train/utt2spk --output /data/train/train.csv
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --device cuda
```

If you have no GPU, `--device cpu` works but training the full model is very slow. On Windows, set `num_workers` to 0 in the YAML if multiprocessing is problematic.

Extracting a test set does not require speaker labels:

```bash
python -m Dataset.prepare --wav-scp /data/eval/wav.scp --output /data/eval/eval.csv
python -m Inference.extract --checkpoint runs/afsc/last.pt --csv /data/eval/eval.csv --output runs/afsc/eval.npz --device cuda
python -m Evaluation.score --enrol runs/afsc/eval.npz --trials /data/eval/trials --output runs/afsc/scores
```

Each line of trials is "enrol audio ID test audio ID label", with 1 for same speaker and 0 for different speaker. Use splits and trials consistent with the actual dataset protocol. The code does not guess the CN-Celeb enrolment aggregation scheme or generate random research trials that replace the official protocol.

Add `--feature fbank` or `--feature mfcc` to the training command to switch baselines, and change the output directory at the same time. Extraction reads the feature type from the checkpoint automatically, so no re-specification is needed.

## Differences from the original code

- `Model/config.yaml`: paper configuration. 2+8 epochs, constant margin 0.3, `channels=[1024,1024,1024,1024,3072]`, 192-d embeddings.
- `Model/legacy_schedule.yaml`: schedule option corresponding to the historical log. 2+10 epochs, margin ramp from 0 to 0.3, learning rate reaching minimum at the end of epoch 10. It expresses a training schedule only and does not prove correspondence to any paper result.
- `Model/demo.yaml`: reduced model for quick CPU validation.
- AFSC preserves the original initialization reference array, positive-gap normalization, and filter formula. Its precise bounds are approximately 20.039–7614.844 Hz, and the first actual output points are affected by normalization. It was not silently replaced with a new "exact 20–7600 Hz initialization".
- The new CSV uses one row per utterance by default, with a random 3-second crop at training time. When the user explicitly provides start/stop, that segment is read first and then cropped. The old "multiple CSV rows but training ignores segment offsets" behavior is not retained.
- The new trainer saves full state for resumption and uses a new checkpoint format. Legacy `embedding_model.ckpt` cannot be loaded directly and must be converted with `tools/convert_legacy.py`.
- Scoring explicitly handles tied scores and ROC endpoints, so it may differ slightly from the old metric function when scores are equal.

## Resume and inspect results

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
python -m Features.export --checkpoint runs/afsc/last.pt --output runs/afsc/filters --plot
```

Training output includes per-epoch logs, configuration, speaker indices, full checkpoints, and frequency points. Resume starts at the next epoch; mid-epoch resume is not supported. Full checkpoints contain data manifest paths, so review those paths before publishing a model.

## Validation

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

See TESTING.md for the current test scope. Passing tests means the software workflow runs; it does not replace training on real data and experimental validation.

## Attribution

See `THIRD_PARTY_NOTICES.md` and `LICENSE`. This release builds on user-provided 3D-Speaker / SpeechBrain-derived ECAPA and preprocessing code; attribution is retained. Cite the AFSC paper when using the proposed feature method. Research audio must be obtained from its original providers.

`tools/convert_legacy.py` is original code for this project and converts 3D-Speaker-style AFSC + ECAPA checkpoints into this project's format. The converted checkpoint contents remain under their original license. If you use the pretrained weights, make sure you have the right to redistribute them.
```
