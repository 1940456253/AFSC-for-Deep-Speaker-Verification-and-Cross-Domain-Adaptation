<a id="top"></a>

# 🎙️ AFSC: Speech Features for Deep Speaker Verification and Cross-Domain Adaptation

**English** | [中文](README_中文.md) | [日本語](README_日本語.md)

**Learnable Acoustic Features · Speaker Verification · Target-Domain Adaptation**

This project provides code for training AFSC features and ECAPA-TDNN, extracting embeddings, speaker verification, and cross-domain adaptation, with MFCC and FBank features retained for comparison. Original reference code for Res2Net and X-Vector is also provided for users to integrate as needed.

> 📢 **Release Updates**
>
> - **Cross-domain experiment code is now public**: Code for SITW target-domain adaptation is available in [`Adaptation/`](Adaptation/) for fine-tuning and comparative experiments.
> - **Pretrained model is now public**: The **ECAPA + AFSC model trained for 10 epochs on CN-Celeb2** is available for download from this project's [GitHub Releases](../../releases).

| Usage | Entry Point | Purpose |
|---|---|---|
| 🏋️ Train from scratch | `Model/train.py` | Train AFSC + ECAPA on example audio or your own dataset |
| 📦 Load a pretrained model | [Releases](../../releases) · `Inference/` | Extract embeddings, compute scores, and compare two recordings |
| 🔄 Target-domain adaptation | [`Adaptation/`](Adaptation/) | Fine-tune a pretrained model across domains and compare scores |

Models trained here and pretrained models share the scripts in `Inference/`, `Evaluation/`, and `Features/`; cross-domain adaptation uses `Adaptation/`.

### 🧭 Quick Navigation

[Project Structure](#structure) · [Installation](#installation) · [Demo Data](#demo) · [Training and Verification](#training) · [Cross-Domain Adaptation](#adaptation) · [Pretrained Model](#pretrained) · [Your Own Data](#data) · [Configuration](#configuration) · [Resume Training](#resume) · [Tests](#tests) · [Citation and Attribution](#credits)

---

<a id="structure"></a>

## 📂 Project Structure

| Directory | Main Files | Function |
|---|---|---|
| Dataset | data.py, prepare.py, demo.py | Read audio, prepare training manifests, and generate demo data from speaker directories |
| Features | spectrum.py, afsc.py, frontend.py, export.py | Power spectra and three acoustic feature types; export learned filters |
| Model | ecapa_tdnn.py, system.py, train.py | ECAPA, classification loss, two-stage training, and resumption |
| Evaluation | metrics.py, score.py, test_pipeline.py, smoke_test.py | Scoring, EER/MinDCF, and validation code |
| Inference | extract.py, verify.py | Extract embeddings and compare two recordings |
| Adaptation | adapter.py, losses.py, chunking.py, standardize.py, trials.py, sampler.py, dataset.py, embed.py, train.py, demo.py | Target-domain adaptation: residual adapter, embedding standardization, fixed fusion, and AFSC gap updates |

Additional directories:

| Directory | Contents |
|---|---|
| `Speech_example/` | Example audio: one speaker per subdirectory, with 15 recordings in each |
| `PreTrained/` | Local storage for pretrained weights downloaded from Releases and the converted `cn_celeb2_afsc_ecapa.pt` |
| `tools/` | Utilities, including `convert_legacy.py` and `inspect_ckpt.py` |

Training code is located in Model, so no separate Training directory is added. Run all commands from the project root.

<a id="installation"></a>

## 🛠️ Installation

Use Python 3.10–3.12. First install matching PyTorch and torchaudio 2.5.1 packages, then install the dependencies in requirements.txt. CPU example:

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU use, install the appropriate CUDA build; no changes to `requirements.txt` are needed.

<a id="demo"></a>

## 🚀 Prepare Demo Data

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

Generated files are stored in `demo_data/`:

| File | Contents |
|---|---|
| `train.csv` | Training manifest; each row contains `ID,path,spk` |
| `eval.csv` | Evaluation manifest; each row contains `ID,path,spk` |
| `trials.txt` | Evaluation trials; each line contains `enrol_id test_id label` |

By default, five recordings per speaker are reserved for evaluation and the remainder are used for training. To change this:

```bash
python -m Dataset.demo --eval-per-speaker 3
```

<a id="training"></a>

## 🏋️ Part 1: Training and Speaker Verification

Open the project root in PyCharm, select a Python interpreter with the dependencies installed, and set the run configuration's working directory to the project root. Run the scripts in order using their default demo settings, or execute the commands below from the command line.

### 1. Train

```bash
python -m Model.train --config Model/demo.yaml --train-csv demo_data/train.csv --output runs/demo --device cpu
```

`Model/demo.yaml` defines a small, CPU-friendly model. `Model/config.yaml` contains the paper's configuration (2+8 epochs, `channels=[1024,1024,1024,1024,3072]`, and 192-dimensional embeddings); use it for full experiments.

Training outputs are stored in `runs/demo/`:

| File | Contents |
|---|---|
| `config.yaml` | Complete configuration used for this training run |
| `speakers.json` | Speaker → index mapping |
| `train.jsonl` | Per-epoch loss / accuracy / lr / margin |
| `epoch_XXX.pt` | Complete checkpoint for each epoch |
| `last.pt` | Copy of the final epoch's checkpoint |
| `frequency_points_XXX.json` | AFSC frequency points for each epoch |

### 2. Extract Evaluation Embeddings

```bash
python -m Inference.extract --checkpoint runs/demo/last.pt --csv demo_data/eval.csv --output runs/demo/embeddings.npz --device cpu
```

Outputs:

- `runs/demo/embeddings.npz`: contains `ids` (a string array) and `embeddings` (N × D)
- `runs/demo/embeddings.json`: metadata recording checkpoint, csv, feature, utterances, and chunk_seconds

### 3. Score

```bash
python -m Evaluation.score --enrol runs/demo/embeddings.npz --trials demo_data/trials.txt --output runs/demo/scores
```

Outputs:

- `runs/demo/scores/scores.csv`: enrol_id, test_id, label, and score for each trial
- `runs/demo/scores/metrics.json`: EER, MinDCF, and the number of trials

EER is reported as a percentage. MinDCF is normalized, with a default target prior of `0.01`.

### 4. Export Filters (Optional)

```bash
python -m Features.export --checkpoint runs/demo/last.pt --output runs/demo/filters --plot
```

Outputs:

- `runs/demo/filters/afsc_filters.npz`: contains `bin_points` (82,), `hz` (82,), and `filters` (80, 257)
- `runs/demo/filters/afsc_filters.png` (when `--plot` is specified)

### 5. Compare Two Recordings (Optional)

```bash
python -m Inference.verify --checkpoint runs/demo/last.pt --enrol Speech_example/speech_demo1/speech-01-001.flac --test Speech_example/speech_demo1/speech-01-002.flac
```

If `--enrol` / `--test` are omitted, the first two recordings in `Speech_example/speech_demo1/` are used by default. Calibrate `--threshold` on a development set; there is no universal default threshold.

<a id="adaptation"></a>

## 🔄 Part 2: Cross-Domain Adaptation and Fine-Tuning

**The SITW cross-domain adaptation experiment code is publicly available in [`Adaptation/`](Adaptation/).** Users can fine-tune using the released code, pretrained models, and their own target-domain data, and adjust the number of training epochs, sampling settings, and AFSC gap update strategy.

This part corresponds to Section 3.5 of the paper. The commands below first demonstrate the workflow using the three speakers in `demo_data/`; full experiments should use the complete SITW Dev / Eval splits and the corresponding official trials.

### Method Components

The main settings used in the paper's experiments are:

- Freeze the pretrained speaker encoder;
- Attach a residual adapter after the paper model's 192-dimensional embedding: `192 → 384 → 192`, with ReLU, Dropout 0.40, zero-initialized fc2, and `alpha` initialized to 0.1;
- Training loss: supervised contrastive loss + 0.05 × identity MSE;
- PK sampling with P=8 / K=4 and 100 batches per epoch in the paper's experiments; sampling settings for a small demo must match the available number of speakers;
- Per-dimension standardization of target-domain embeddings, followed by L2 normalization;
- Score fusion with fixed 0.5/0.5 weights;
- AFSC features additionally support target-domain gap updates (2 epochs by default in the demo).

### Quick Demo

The following demo uses the outputs from Part 1:

```text
runs/demo/last.pt
demo_data/train.csv
demo_data/eval.csv
```

Run the demo:

```bash
python -m Adaptation.demo --device cpu
```

To run only the adapter without AFSC gap updates:

```bash
python -m Adaptation.train --device cpu --gap-epochs 0
```

Full workflow (2 epochs of AFSC gap updates by default):

```bash
python -m Adaptation.train --device cpu
```

Outputs are stored in `runs/demo_adapt/` (or `runs/demo_train/`):

| File | Contents |
|---|---|
| `summary.json` | EER and MinDCF for baseline / standardized / adapter / fusion scores |
| `history.json` | Per-epoch adapter loss / supcon / identity / alpha |
| `gap_history.json` | AFSC gap update history; generated only when `--gap-epochs > 0` |

The four sections in `summary.json` mean:

| Field | Meaning | Corresponding Column in the Paper |
|---|---|---|
| `baseline` | Unadapted cosine scores computed directly from pretrained embeddings | Before |
| `standardized` | Scores after target-domain per-dimension standardization and L2 normalization | Standardized |
| `adapter` | Scores from the residual adapter branch; when gap updates are enabled, these also reflect front-end adaptation | Adapter |
| `fusion` | `0.5 × adapter + 0.5 × standardized` | Fusion |

> 💡 **Demo vs. Full Experiments**
>
> The demo uses a small model (`Model/demo.yaml`) and a small amount of audio; its results do not represent the paper's reported values. For full cross-domain experiments, use the CN-Celeb2 pretrained model from Releases and configure SITW data paths, training parameters, and evaluation protocols according to the code in `Adaptation/`.

### Run SITW or Custom Target-Domain Experiments

1. Download the pretrained weights and prepare a supported checkpoint as described in the next section.
2. Prepare target-domain training and evaluation manifests, along with trials consistent with the evaluation protocol.
3. Configure model paths, data paths, and fine-tuning parameters in `Adaptation/`.
4. Compare EER / MinDCF for unadapted, standardized, adapter, and fused scores.

SITW experiments use Dev data for adaptation and parameter selection, and Eval for final evaluation. Estimate standardization statistics from the target-domain adaptation data. Example data is only for learning how to run the code and cannot replace a formal cross-domain evaluation protocol.

<a id="pretrained"></a>

## 📦 Download and Load the Pretrained Model

### ⬇️ Get the Model

**The ECAPA + AFSC model pretrained for 10 epochs on CN-Celeb2 is publicly available for download from this project's [Releases page](../../releases).**

Open Releases and download the weights or archive from **Assets** under the appropriate release. Extract the archive if needed and place the model files in `PreTrained/` at the project root.

| Item | Description |
|---|---|
| Model | ECAPA-TDNN + AFSC |
| Pretraining dataset | CN-Celeb2 |
| Training duration | 10 epochs |
| Embedding dimension | 192 |
| Download | [This project's GitHub Releases](../../releases) |
| Local storage | `PreTrained/` |

### Convert Legacy Weights

If you downloaded legacy checkpoints, arrange the files as follows:

```
PreTrained/embedding_model.ckpt    # AFSC + ECAPA weights
PreTrained/classifier.ckpt         # 2793 × 192 classification head
```

First convert the legacy files to this project's `format_version: 1` using `tools/convert_legacy.py`:

```bash
python tools/convert_legacy.py
```

The conversion script:

1. Changes the `afsc.*` prefix to `frontend.*`;
2. Changes the `ecapa.*` prefix to `encoder.*`;
3. Checks that all keys match `SpeakerSystem`;
4. Writes `PreTrained/cn_celeb2_afsc_ecapa.pt` and `PreTrained/cn_celeb2_afsc_ecapa.json`.

The message `All keys matched exactly.` indicates that the weight keys matched successfully.

> 💡 If the download already includes the converted `cn_celeb2_afsc_ecapa.pt`, use it directly without converting again.

### Extract and Score with the Pretrained Model

**Command-line usage:** Specify the model and output paths directly through arguments, without changing the scripts' defaults:

```bash
python -m Inference.extract --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --csv demo_data/eval.csv --output runs/pretrained_eval.npz --device cpu
python -m Evaluation.score --enrol runs/pretrained_eval.npz --trials demo_data/trials.txt --output runs/pretrained_scores
python -m Features.export --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --output runs/pretrained_filters --plot
```

When running directly in PyCharm without arguments, follow the comments in sections (A)/(B) of each script to switch `default_checkpoint()` or `default_paths()`, keeping model, embedding, and output paths consistent.

### Usage Notes

- EER on a small demo is sensitive to sample size and recording conditions and should not be used to judge the pretrained model's overall performance. Speaker verification inherently targets unseen speakers, so evaluation errors should not be attributed solely to the speakers being absent from training.
- Meaningful evaluation requires the official CN-Celeb2 or SITW evaluation protocols and trial lists.
- The pretrained model was trained with an AFSC front end and can only be used with `feature: afsc`.
- The speaker mapping stored in the checkpoint uses placeholder names (`speaker_00000`…`speaker_02792`); this only affects mapping classification outputs back to actual speaker names.

<a id="data"></a>

## 🗂️ Use Your Own Dataset

Prepare 16 kHz audio and the training set's wav.scp and utt2spk files. Each wav.scp line contains “audio ID path”; each utt2spk line contains “audio ID speaker ID”. Audio files are not copied into the repository.

```bash
python -m Dataset.prepare --wav-scp /data/train/wav.scp --utt2spk /data/train/utt2spk --output /data/train/train.csv
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --device cuda
```

Without a GPU, use `--device cpu`, but training the full model will be slow. If multiprocessing causes issues on Windows, set `num_workers` to 0 in the YAML file.

Speaker labels are not required to extract test-set embeddings:

```bash
python -m Dataset.prepare --wav-scp /data/eval/wav.scp --output /data/eval/eval.csv
python -m Inference.extract --checkpoint runs/afsc/last.pt --csv /data/eval/eval.csv --output runs/afsc/eval.npz --device cuda
python -m Evaluation.score --enrol runs/afsc/eval.npz --trials /data/eval/trials --output runs/afsc/scores
```

Each trials line contains “enrollment audio ID test audio ID label”: 1 for the same speaker and 0 for different speakers. Use splits and trials that follow the actual dataset protocol. The code does not infer CN-Celeb enrollment aggregation rules or generate random research test sets as substitutes for official protocols.

Add `--feature fbank` or `--feature mfcc` to the training command to switch baselines, and use a different output directory. During extraction, the feature type is read automatically from the checkpoint and does not need to be specified again.

Cross-domain adaptation accepts the same CSV format. `Adaptation.train` and `Adaptation.demo` read manifests through `--train-csv` / `--eval-csv`, select the pretrained model through `--checkpoint`, and control long-audio processing through `--chunk-seconds` (`None` processes the entire recording; 30.0 uses 30-second chunks).

<a id="configuration"></a>

## ⚙️ Configuration and Implementation Notes

- `Model/config.yaml`: the paper's configuration, with 2+8 epochs, margin fixed at 0.3, `channels=[1024,1024,1024,1024,3072]`, and 192-dimensional embeddings.
- `Model/legacy_schedule.yaml`: retains the training schedule option corresponding to the logs: 2+10 epochs, margin increasing from 0 to 0.3, and the learning rate reaching its minimum at the end of epoch 10. It describes a training schedule only and does not establish correspondence to any specific paper result.
- `Model/demo.yaml`: a reduced model for quick CPU-based checks.
- AFSC retains the original initialization reference array, positive-gap normalization, and filtering formulas. Its exact boundaries are approximately 20.039–7614.844 Hz, and normalization also affects the first frequency points actually produced. This initialization is preserved rather than resetting it to exactly 20–7600 Hz.
- The new CSV format uses one row per recording by default, with random 3-second cropping during training. If start/stop values are explicitly provided, the specified interval is read before cropping. The old behavior of splitting recordings across CSV rows while ignoring those intervals during training is not retained.
- The new trainer saves the full state for resuming training and uses a new checkpoint format. Legacy `embedding_model.ckpt` files cannot be loaded directly and must first be converted with `tools/convert_legacy.py`.
- Scoring explicitly handles tied scores and ROC endpoints, so results may differ slightly from the old metric functions when scores are tied.
- The residual adapter in `Adaptation/` follows Section 3.5.2 of the paper. Identity loss uses a detached base embedding as its reference target, preventing gradients from passing through the reference-target branch.
- During AFSC gap updates, the encoder stays in `eval()` mode; only `raw_gaps` and the adapter are updated. Res2Net and X-Vector do not receive additional gap updates in the paper.

<a id="resume"></a>

## 💾 Resume Training and Inspect Results

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
python -m Features.export --checkpoint runs/afsc/last.pt --output runs/afsc/filters --plot
```

Training outputs include per-epoch logs, configuration, speaker indices, complete checkpoints, and frequency points. Training resumes at the next epoch; mid-epoch resumption is not supported. Full checkpoints contain data manifest paths, which should be reviewed before releasing a model.

<a id="tests"></a>

## 🧪 Software Tests

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

See TESTING.md for the current test coverage. Passing tests indicates that the software workflow runs, but does not replace training and experimental validation on real datasets.

<a id="credits"></a>

## 📚 Citation and Attribution

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSE`](LICENSE) for code sources and licensing. This project is organized around ECAPA and preprocessing code derived from 3D-Speaker / SpeechBrain, with the relevant attribution retained. Please cite the AFSC paper when using the proposed feature method. Audio for research must be obtained from its original provider.

`tools/convert_legacy.py` is original code developed for this project to convert 3D-Speaker-style AFSC + ECAPA checkpoints to this project's format. Converted checkpoint contents remain subject to their original licenses. When using pretrained weights, ensure that you have redistribution rights.

The residual adapter, supervised contrastive loss, target-domain embedding standardization, and fixed score fusion in `Adaptation/` follow Section 3.5 of the paper and the experimental notes provided by the author. The corresponding SITW cross-domain adaptation experiment code is publicly available in `Adaptation/`; use it to configure your data and run fine-tuning experiments.

### Demo Dataset

`Speech_example/` contains a small number of recordings from CN-Celeb (OpenSLR SLR82), used only to demonstrate the workflow; this repository does not include the full dataset.

- Dataset homepage: https://openslr.org/82/
- License: Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- Source: released by CSLT (Center for Speech and Language Technologies), Tsinghua University

CN-Celeb:

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

When using the example audio above, comply with its CC BY-SA 4.0 license and retain the attribution. For code licensing, see this repository's `LICENSE` and `THIRD_PARTY_NOTICES.md`. This repository does not distribute the complete CN-Celeb dataset, only a small amount of example audio. Obtain the full dataset from https://openslr.org/82/.


---

[⬆️ Back to Top](#top)
