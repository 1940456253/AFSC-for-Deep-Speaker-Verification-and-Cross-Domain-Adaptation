# AFSC + ECAPA 第一部分公开代码

[English](README.md) | **中文** | [日本語](README.ja.md)

这个项目覆盖论文第一部分的 AFSC 特征、ECAPA 训练、嵌入提取和说话人验证，保留 MFCC、FBank 对照。没有加入跨域适应，也没有加入 Res2Net 或 X-Vector。代码可以重新训练，也支持加载论文提供的 CN-Celeb2 预训练模型。

项目同时支持两种使用方式：

- **自己训练**：从 `Speech_example/` 或你自己的数据集出发，用 `Model/train.py` 训练 AFSC + ECAPA；
- **加载预训练**：用 `tools/convert_legacy.py` 把论文提供的 `embedding_model.ckpt` + `classifier.ckpt` 转换成本项目的 checkpoint 格式，然后直接提取、评分、对比。

两种方式共用同一套 `Inference/`、`Evaluation/`、`Features/` 脚本。

## 五个主要目录

| 文件夹 | 主要文件 | 功能 |
|---|---|---|
| Dataset | data.py、prepare.py、demo.py | 读取音频、准备训练清单、从说话人文件夹生成 demo 数据 |
| Features | spectrum.py、afsc.py、frontend.py、export.py | 功率谱与三种声学特征、导出学习后的滤波器 |
| Model | ecapa_tdnn.py、system.py、train.py | ECAPA、分类损失、两阶段训练与恢复 |
| Evaluation | metrics.py、score.py、test_pipeline.py、smoke_test.py | 评分、EER/MinDCF、验证代码 |
| Inference | extract.py、verify.py | 提取嵌入、比较两段音频 |

除此之外还有：

| 文件夹 | 内容 |
|---|---|
| `Speech_example/` | 示例音频，每子文件夹一位说话人，每文件夹内 15 条语音 |
| `PreTrained/` | 论文提供的预训练 checkpoint（`embedding_model.ckpt` + `classifier.ckpt`）和转换后的 `cn_celeb2_afsc_ecapa.pt` |
| `tools/` | 辅助脚本，包括 `convert_legacy.py` 和 `inspect_ckpt.py` |

训练代码放在 Model 中，因此不额外增加第六个 Training 目录。所有命令都从项目根目录运行。

## 安装

使用 Python 3.10–3.12。先安装相匹配的 PyTorch 与 torchaudio 2.5.1，再安装 requirements.txt。CPU 版本示例：

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

GPU 用户安装对应的 CUDA 版本即可，`requirements.txt` 不用改。


## 先跑通演示

从 `Speech_example/` 生成清单：

```bash
python -m Dataset.demo
```

预期输出：

```
Train:  30 utterances from 3 speakers
Eval:   15 utterances from 3 speakers
Trials: 105 (30 target, 75 nontarget)
Manifests written to .../demo_data
```

生成的文件在 `demo_data/`：

| 文件 | 内容 |
|---|---|
| `train.csv` | 训练清单，每行 `ID,path,spk` |
| `eval.csv` | 评估清单，每行 `ID,path,spk` |
| `trials.txt` | 评估 trial，每行 `enrol_id test_id label` |

默认每个说话人保留 5 条用于评估，其余用于训练。可以改：

```bash
python -m Dataset.demo --eval-per-speaker 3
```

## 自己训练的完整流程

在 PyCharm 里依次右键 Run 以下文件；或者从命令行用 `python -m ...` 依次运行。

### 1. 训练

```bash
python -m Model.train --config Model/demo.yaml --train-csv demo_data/train.csv --output runs/demo --device cpu
```

`Model/demo.yaml` 是 CPU 友好的小模型；`Model/config.yaml` 是论文配置（2+8 epoch、`channels=[1024,1024,1024,1024,3072]`、192 维嵌入），正式实验用这个。

训练输出在 `runs/demo/`：

| 文件 | 内容 |
|---|---|
| `config.yaml` | 本次训练使用的完整配置 |
| `speakers.json` | 说话人 → 索引映射 |
| `train.jsonl` | 每轮的 loss / accuracy / lr / margin |
| `epoch_XXX.pt` | 每个 epoch 的完整 checkpoint |
| `last.pt` | 最后一个 epoch 的副本 |
| `frequency_points_XXX.json` | 每个 epoch 的 AFSC 频率点 |

### 2. 提取评估集嵌入

```bash
python -m Inference.extract --checkpoint runs/demo/last.pt --csv demo_data/eval.csv --output runs/demo/embeddings.npz --device cpu
```

输出：

- `runs/demo/embeddings.npz`：含 `ids`（字符串数组）和 `embeddings`（N × D）
- `runs/demo/embeddings.json`：元数据，记录 checkpoint、csv、feature、utterances、chunk_seconds

### 3. 评分

```bash
python -m Evaluation.score --enrol runs/demo/embeddings.npz --trials demo_data/trials.txt --output runs/demo/scores
```

输出：

- `runs/demo/scores/scores.csv`：每个 trial 的 enrol_id、test_id、label、score
- `runs/demo/scores/metrics.json`：EER、MinDCF、trial 数量

EER 的单位是百分比，MinDCF 使用归一化值，默认目标先验为 `0.01`。

### 4. 导出滤波器（可选）

```bash
python -m Features.export --checkpoint runs/demo/last.pt --output runs/demo/filters --plot
```

输出：

- `runs/demo/filters/afsc_filters.npz`：含 `bin_points` (82,)、`hz` (82,)、`filters` (80, 257)
- `runs/demo/filters/afsc_filters.png`（如果加了 `--plot`）

### 5. 比较两段音频（可选）

```bash
python -m Inference.verify --checkpoint runs/demo/last.pt --enrol Speech_example/speech_demo1/speech-01-001.flac --test Speech_example/speech_demo1/speech-01-002.flac
```

不传 `--enrol` / `--test` 时，默认使用 `Speech_example/speech_demo1/` 下前两条音频。`--threshold` 需要在开发集上校准，没有通用默认值。

## 加载预训练模型

论文作者提供了 CN-Celeb2 训练的 AFSC + ECAPA 权重：

```
PreTrained/embedding_model.ckpt    # AFSC + ECAPA 权重
PreTrained/classifier.ckpt         # 2793 × 192 的分类头
```

这两个是旧版格式，需要先用 `tools/convert_legacy.py` 转换成本项目的 `format_version: 1`：

```bash
python tools/convert_legacy.py
```

转换脚本会：

1. 把 `afsc.*` 前缀改成 `frontend.*`；
2. 把 `ecapa.*` 前缀改成 `encoder.*`；
3. 校验所有 key 都能匹配到 `SpeakerSystem`；
4. 写出 `PreTrained/cn_celeb2_afsc_ecapa.pt` 和 `PreTrained/cn_celeb2_afsc_ecapa.json`。

如果看到 `All keys matched exactly.`，说明转换完全成功。

### 用预训练模型提取、评分

切换方式：打开 `Inference/extract.py`、`Evaluation/score.py`、`Features/export.py`、`Inference/verify.py`，把 `default_checkpoint()`（或 `default_paths()`）里 (A) 段注释掉、(B) 段取消注释。

之后就能用和「自己训练」一样的命令：

```bash
python -m Inference.extract --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --csv demo_data/eval.csv --output runs/pretrained_eval.npz --device cpu
python -m Evaluation.score --enrol runs/pretrained_eval.npz --trials demo_data/trials.txt --output runs/pretrained_scores
python -m Features.export --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --output runs/pretrained_filters --plot
```

### 预训练模型需要注意的点

- `Speech_example/` 里的 3 位说话人**不在** CN-Celeb2 的 2793 位说话人中。预训练模型没见过他们，所以在这个 demo 上 EER 可能偏高，这是正常的。
- 真正有意义的评估要用 CN-Celeb2 或 SITW 的官方评估协议和 trial 列表。
- 预训练模型是 AFSC 前端训练的，所以只能配 `feature: afsc` 使用。
- checkpoint 里保存的说话人映射是占位符（`speaker_00000`…`speaker_02792`），只影响把分类结果映射回真实说话人名。

## 换成实际数据

准备 16 kHz 音频，以及训练集的 wav.scp、utt2spk。wav.scp 每行是「音频 ID 路径」，utt2spk 每行是「音频 ID 说话人 ID」。音频文件不复制进仓库。

```bash
python -m Dataset.prepare --wav-scp /data/train/wav.scp --utt2spk /data/train/utt2spk --output /data/train/train.csv
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --device cuda
```

没有 GPU 时可使用 `--device cpu`，但正式模型训练会很慢。Windows 下如果遇到多进程问题，将 YAML 中 `num_workers` 设为 0。

提取测试集不需要说话人标签：

```bash
python -m Dataset.prepare --wav-scp /data/eval/wav.scp --output /data/eval/eval.csv
python -m Inference.extract --checkpoint runs/afsc/last.pt --csv /data/eval/eval.csv --output runs/afsc/eval.npz --device cuda
python -m Evaluation.score --enrol runs/afsc/eval.npz --trials /data/eval/trials --output runs/afsc/scores
```

trials 每行是「注册音频 ID 测试音频 ID 标签」，同一人为 1，不同人为 0。请使用与实际数据集协议相符的划分和 trials。代码不会自行猜测 CN-Celeb 的注册集聚合方式，也不会生成替代官方协议的随机研究测试集。

在训练命令中增加 `--feature fbank` 或 `--feature mfcc` 即可切换基线，同时更换输出目录。提取时会自动从 checkpoint 读取特征类型，不需要重新指定。

## 配置与原有代码的区别

- `Model/config.yaml`：论文配置，2+8 轮，margin 固定为 0.3，`channels=[1024,1024,1024,1024,3072]`，嵌入维度 192。
- `Model/legacy_schedule.yaml`：保留日志对应的训练时序选项，2+10 轮、margin 从 0 增长到 0.3，学习率在第 10 轮末达到最小值。它只能表达训练日程，不能证明对应了某个论文结果。
- `Model/demo.yaml`：缩小模型、CPU 快速验证专用。
- AFSC 保留原始代码的初始化参考数组、正间隔归一化和滤波公式。其精确边界约为 20.039–7614.844 Hz，首次实际输出的点也会受归一化影响。没有悄悄将它替换成新的「精确 20–7600 Hz 初始化」。
- 新 CSV 默认每条音频一行，在训练时随机裁剪 3 秒。用户主动提供 start/stop 时，会先按该区间读取，再裁剪。原代码「CSV 多行分段但训练忽略区间」的行为没有继续沿用。
- 新训练器保存完整状态以恢复训练，采用新的 checkpoint 格式。旧的 `embedding_model.ckpt` 不能直接加载，必须先用 `tools/convert_legacy.py` 转换。
- 评分显式处理相同分数和 ROC 端点，所以在分数相同的情况下可能与旧指标函数有细微差别。

## 训练恢复及查看结果

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
python -m Features.export --checkpoint runs/afsc/last.pt --output runs/afsc/filters --plot
```

训练输出包括逐轮日志、配置、说话人编号、完整 checkpoint 和频率点。恢复到下一轮，不支持轮中间恢复。完整 checkpoint 内含数据清单路径，准备公开模型时应检查这些路径。

## 检查程序

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

当前测试范围见 TESTING.md。测试通过说明软件流程可运行，不能代替真实数据集训练和实验验证。

## 归因

见 `THIRD_PARTY_NOTICES.md` 和 `LICENSE`。这个 release 建立在用户提供的 3D-Speaker / SpeechBrain 派生 ECAPA 和预处理代码之上，归属保留。使用论文提出的特征方法时请引用 AFSC 论文。研究用音频必须从原始提供方获取。

`tools/convert_legacy.py` 是本项目原创代码，用于把 3D-Speaker 风格的 AFSC + ECAPA checkpoint 转换成本项目格式。转换后的 checkpoint 内容仍受原始许可约束。使用预训练权重时请确认你有再分发权限。
```

用户下次上传代码时我再核对文件名引用是否全部同步。你先把这个中文 README 复制替换掉 `README_zh.md`，然后告诉我，我再输出英文和日语的对应版本。
