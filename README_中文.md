<a id="top"></a>

# 🎙️ AFSC：用于深度说话人验证与跨域自适应的语音特征 

[English](README.md) | **中文** | [日本語](README_日本語.md)

**可学习声学特征 · 说话人验证 · 目标域自适应**

本项目提供 AFSC 特征与 ECAPA-TDNN 的训练、嵌入提取、说话人验证和跨域自适应代码，同时保留 MFCC、FBank 特征作为对照。Res2Net 和 X-Vector 提供原始参考代码，用户可根据需要自行接入。

> 📢 **公开内容更新**
>
> - **跨域实验代码已公开**：SITW 目标域自适应相关代码已放入 [`Adaptation/`](Adaptation/)，可根据代码开展微调和对比实验。
> - **预训练模型已公开**：在 **CN-Celeb2 上训练 10 个 epoch 的 ECAPA + AFSC 模型**已发布至本项目的 [GitHub Releases](../../releases)，可下载使用。

| 使用方式 | 入口 | 适用场景 |
|---|---|---|
| 🏋️ 从头训练 | `Model/train.py` | 使用示例音频或自己的数据集训练 AFSC + ECAPA |
| 📦 加载预训练模型 | [Releases](../../releases) · `Inference/` | 提取嵌入、评分及比较两段音频 |
| 🔄 目标域自适应 | [`Adaptation/`](Adaptation/) | 基于预训练模型开展跨域微调与评分对比 |

训练和预训练模型共用 `Inference/`、`Evaluation/`、`Features/` 脚本；跨域自适应使用 `Adaptation/` 目录。

### 🧭 快速导航

[项目结构](#structure) · [环境安装](#installation) · [演示数据](#demo) · [训练与验证](#training) · [跨域自适应](#adaptation) · [预训练模型](#pretrained) · [实际数据](#data) · [配置说明](#configuration) · [恢复训练](#resume) · [测试](#tests) · [引用与归因](#credits)

---

<a id="structure"></a>

## 📂 项目结构

| 文件夹 | 主要文件 | 功能 |
|---|---|---|
| Dataset | data.py、prepare.py、demo.py | 读取音频、准备训练清单、从说话人文件夹生成 demo 数据 |
| Features | spectrum.py、afsc.py、frontend.py、export.py | 功率谱与三种声学特征、导出学习后的滤波器 |
| Model | ecapa_tdnn.py、system.py、train.py | ECAPA、分类损失、两阶段训练与恢复 |
| Evaluation | metrics.py、score.py、test_pipeline.py、smoke_test.py | 评分、EER/MinDCF、验证代码 |
| Inference | extract.py、verify.py | 提取嵌入、比较两段音频 |
| Adaptation | adapter.py、losses.py、chunking.py、standardize.py、trials.py、sampler.py、dataset.py、embed.py、train.py、demo.py | 目标域自适应：residual adapter、embedding 标准化、固定融合、AFSC gap 更新 |

除此之外还有：

| 文件夹 | 内容 |
|---|---|
| `Speech_example/` | 示例音频，每子文件夹一位说话人，每文件夹内 15 条语音 |
| `PreTrained/` | 从 Releases 下载的预训练权重及转换后的 `cn_celeb2_afsc_ecapa.pt` 的本地存放目录 |
| `tools/` | 辅助脚本，包括 `convert_legacy.py` 和 `inspect_ckpt.py` |

训练代码放在 Model 中，因此不额外增加 Training 目录。所有命令都从项目根目录运行。

<a id="installation"></a>

## 🛠️ 环境安装

使用 Python 3.10–3.12。先安装相匹配的 PyTorch 与 torchaudio 2.5.1，再安装 requirements.txt。CPU 版本示例：

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

GPU 用户安装对应的 CUDA 版本即可，`requirements.txt` 不用改。

<a id="demo"></a>

## 🚀 准备演示数据

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

<a id="training"></a>

## 🏋️ 第一部分：训练与说话人验证

在 PyCharm 中以项目根目录打开工程，选择已安装依赖的 Python 解释器，并将运行配置的工作目录设为项目根目录。可使用脚本中的默认演示配置依次运行，也可在命令行执行下列命令。

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

<a id="adaptation"></a>

## 🔄 第二部分：跨域自适应与微调

**SITW 跨域自适应实验代码已公开在 [`Adaptation/`](Adaptation/) 目录中。** 用户可以基于公开代码、预训练模型和自己的目标域数据开展微调实验，并调整训练轮数、采样设置及 AFSC gap 更新策略。

该部分对应论文 3.5 节。下面的命令先使用 `demo_data/` 的三位说话人演示流程；正式实验应使用完整的 SITW Dev / Eval 划分和对应的官方 trials。

### 方法组成

论文实验的主要设置包括：

- 冻结预训练 speaker encoder；
- 在论文模型的 192 维嵌入后接 residual adapter：`192 → 384 → 192`，ReLU、Dropout 0.40、fc2 零初始化、`alpha` 初始 0.1；
- 训练损失：supervised contrastive loss + 0.05 × identity MSE；
- 论文实验采用 P=8 / K=4 的 PK 采样，每 epoch 100 个 batch；小规模 demo 的采样设置须与实际说话人数匹配；
- target-domain embedding 逐维标准化，再 L2 归一化；
- 固定权重 0.5/0.5 的分数融合；
- AFSC 特征额外支持 target-domain gap 更新（demo 默认 2 epoch）。

### 快速演示

以下演示使用第一部分生成的产物：

```text
runs/demo/last.pt
demo_data/train.csv
demo_data/eval.csv
```

运行演示：

```bash
python -m Adaptation.demo --device cpu
```

如果只想跑 adapter，不触发 AFSC gap 更新：

```bash
python -m Adaptation.train --device cpu --gap-epochs 0
```

完整流程（AFSC 默认 2 epoch gap 更新）：

```bash
python -m Adaptation.train --device cpu
```

输出在 `runs/demo_adapt/`（或 `runs/demo_train/`）：

| 文件 | 内容 |
|---|---|
| `summary.json` | baseline / standardized / adapter / fusion 四种分数的 EER、MinDCF |
| `history.json` | adapter 逐 epoch 的 loss / supcon / identity / alpha |
| `gap_history.json` | 仅当 `--gap-epochs > 0` 时生成，记录 AFSC gap 更新过程 |

`summary.json` 中四个块的含义：

| 字段 | 含义 | 论文对应列 |
|---|---|---|
| `baseline` | 未适配，直接用预训练 embedding 的余弦分数 | Before |
| `standardized` | target-domain 逐维标准化 + L2 后的分数 | Standardized |
| `adapter` | residual adapter 分支的分数；启用 gap 更新时也包含前端适配的影响 | Adapter |
| `fusion` | `0.5 × adapter + 0.5 × standardized` | Fusion |

> 💡 **演示与正式实验**
>
> 演示使用小模型（`Model/demo.yaml`）和少量音频，结果不代表论文数值。正式跨域实验可使用 Releases 中的 CN-Celeb2 预训练模型，并按 `Adaptation/` 中的代码配置 SITW 数据路径、训练参数和评估协议。

### 开展 SITW 或自定义目标域实验

1. 下载预训练权重，并按下一节准备项目支持的 checkpoint。
2. 准备目标域训练和评估清单，以及与评估协议一致的 trials。
3. 在 `Adaptation/` 中配置模型路径、数据路径和微调参数。
4. 对比未适配、标准化、adapter 和融合分数的 EER / MinDCF。

SITW 实验使用 Dev 数据进行适配与参数选择，Eval 用于最终评估。标准化统计量应从目标域适配数据中估计。示例数据仅用于熟悉运行方式，不能替代正式的跨域评估协议。

<a id="pretrained"></a>

## 📦 下载与加载预训练模型

### ⬇️ 获取模型

**ECAPA + AFSC 在 CN-Celeb2 数据集上训练 10 个 epoch 的预训练模型已公开，可从本项目的 [Releases 页面](../../releases) 下载。**

打开 Releases，在对应版本的 **Assets** 中下载权重文件或压缩包，解压后将模型文件放入项目根目录的 `PreTrained/`。

| 项目 | 说明 |
|---|---|
| 模型 | ECAPA-TDNN + AFSC |
| 预训练数据集 | CN-Celeb2 |
| 训练轮数 | 10 epoch |
| 嵌入维度 | 192 |
| 下载位置 | [本项目 GitHub Releases](../../releases) |
| 本地存放位置 | `PreTrained/` |

### 转换旧版权重

若下载的是旧版 checkpoint，将文件放置为：

```
PreTrained/embedding_model.ckpt    # AFSC + ECAPA 权重
PreTrained/classifier.ckpt         # 2793 × 192 的分类头
```

旧版文件需要先用 `tools/convert_legacy.py` 转换成本项目的 `format_version: 1`：

```bash
python tools/convert_legacy.py
```

转换脚本会：

1. 把 `afsc.*` 前缀改成 `frontend.*`；
2. 把 `ecapa.*` 前缀改成 `encoder.*`；
3. 校验所有 key 都能匹配到 `SpeakerSystem`；
4. 写出 `PreTrained/cn_celeb2_afsc_ecapa.pt` 和 `PreTrained/cn_celeb2_afsc_ecapa.json`。

如果看到 `All keys matched exactly.`，说明权重键名匹配成功。

> 💡 如果下载包已包含转换后的 `cn_celeb2_afsc_ecapa.pt`，可以直接使用该文件，无需重复转换。

### 用预训练模型提取、评分

**命令行使用：**直接通过参数指定模型和输出路径，无需修改脚本默认值：

```bash
python -m Inference.extract --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --csv demo_data/eval.csv --output runs/pretrained_eval.npz --device cpu
python -m Evaluation.score --enrol runs/pretrained_eval.npz --trials demo_data/trials.txt --output runs/pretrained_scores
python -m Features.export --checkpoint PreTrained/cn_celeb2_afsc_ecapa.pt --output runs/pretrained_filters --plot
```

若在 PyCharm 中直接运行且不传参数，可按各脚本中 (A)/(B) 段的注释切换 `default_checkpoint()` 或 `default_paths()`，使模型路径、嵌入路径和输出路径相互对应。

### 使用说明

- 小规模 demo 的 EER 容易受样本数量和音频条件影响，不能据此判断预训练模型的整体性能。说话人验证本身面向未见说话人，不应仅以“训练时未见过这些说话人”解释评估误差。
- 真正有意义的评估要用 CN-Celeb2 或 SITW 的官方评估协议和 trial 列表。
- 预训练模型是 AFSC 前端训练的，所以只能配 `feature: afsc` 使用。
- checkpoint 里保存的说话人映射是占位符（`speaker_00000`…`speaker_02792`），只影响把分类结果映射回真实说话人名。

<a id="data"></a>

## 🗂️ 使用自己的数据集

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

跨域自适应部分也接受同样的 CSV 格式。`Adaptation.train` 和 `Adaptation.demo` 通过 `--train-csv` / `--eval-csv` 读取清单，通过 `--checkpoint` 指定预训练模型，通过 `--chunk-seconds` 控制长音频处理（`None` 表示整段处理，30.0 表示 30 秒 chunk）。

<a id="configuration"></a>

## ⚙️ 配置与实现说明

- `Model/config.yaml`：论文配置，2+8 轮，margin 固定为 0.3，`channels=[1024,1024,1024,1024,3072]`，嵌入维度 192。
- `Model/legacy_schedule.yaml`：保留日志对应的训练时序选项，2+10 轮、margin 从 0 增长到 0.3，学习率在第 10 轮末达到最小值。它只能表达训练日程，不能证明对应了某个论文结果。
- `Model/demo.yaml`：缩小模型、CPU 快速验证专用。
- AFSC 保留原始代码的初始化参考数组、正间隔归一化和滤波公式。其精确边界约为 20.039–7614.844 Hz，首次实际输出的点也会受归一化影响。实现沿用这一初始化，而非重新设定为精确的 20–7600 Hz。
- 新 CSV 默认每条音频一行，在训练时随机裁剪 3 秒。用户主动提供 start/stop 时，会先按该区间读取，再裁剪。原代码「CSV 多行分段但训练忽略区间」的行为没有继续沿用。
- 新训练器保存完整状态以恢复训练，采用新的 checkpoint 格式。旧的 `embedding_model.ckpt` 不能直接加载，必须先用 `tools/convert_legacy.py` 转换。
- 评分显式处理相同分数和 ROC 端点，所以在分数相同的情况下可能与旧指标函数有细微差别。
- `Adaptation/` 的 residual adapter 参考论文 3.5.2 节；identity loss 使用 detached base embedding 作为参考目标，使梯度不经过参考目标分支。
- AFSC gap 更新阶段，encoder 保持 `eval()`，只更新 `raw_gaps` 和 adapter；Res2Net 与 X-Vector 在论文中不进行额外的 gap 更新。

<a id="resume"></a>

## 💾 恢复训练与查看结果

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
python -m Features.export --checkpoint runs/afsc/last.pt --output runs/afsc/filters --plot
```

训练输出包括逐轮日志、配置、说话人编号、完整 checkpoint 和频率点。恢复到下一轮，不支持轮中间恢复。完整 checkpoint 内含数据清单路径，准备公开模型时应检查这些路径。

<a id="tests"></a>

## 🧪 软件测试

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

当前测试范围见 TESTING.md。测试通过说明软件流程可运行，不能代替真实数据集训练和实验验证。

<a id="credits"></a>

## 📚 引用与归因

代码来源与许可见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 和 [`LICENSE`](LICENSE)。本项目基于 3D-Speaker / SpeechBrain 派生的 ECAPA 与预处理代码整理，保留相关归属声明。使用论文提出的特征方法时请引用 AFSC 论文。研究用音频必须从原始提供方获取。

`tools/convert_legacy.py` 是本项目原创代码，用于把 3D-Speaker 风格的 AFSC + ECAPA checkpoint 转换成本项目格式。转换后的 checkpoint 内容仍受原始许可约束。使用预训练权重时请确认你有再分发权限。

`Adaptation/` 中的 residual adapter、supervised contrastive loss、target-domain embedding 标准化和固定分数融合均参考论文 3.5 节和作者提供的实验笔记。相关 SITW 跨域自适应实验代码已在 `Adaptation/` 中公开，可据此配置数据并开展微调实验。

### 演示数据集

`Speech_example/` 中包含少量来自 CN-Celeb 数据集（OpenSLR SLR82）的音频，仅用于演示流程；本仓库不包含完整数据集。

- 数据集主页：https://openslr.org/82/
- 许可证：Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- 来源：由清华大学 CSLT（Center for Speech and Language Technologies）发布

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

使用上述演示音频时，请遵守其 CC BY-SA 4.0 许可证，并保留归属声明。代码许可请参阅本仓库的 `LICENSE` 和 `THIRD_PARTY_NOTICES.md`。本仓库不分发完整的 CN-Celeb 数据集，仅包含少量演示音频。如需完整数据集，请从 https://openslr.org/82/ 获取。


---

[⬆️ 返回顶部](#top)
