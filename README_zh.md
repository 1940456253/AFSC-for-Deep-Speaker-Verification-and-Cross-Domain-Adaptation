# AFSC + ECAPA 第一部分公开代码

这个项目覆盖论文第一部分的 AFSC 特征、ECAPA 训练、嵌入提取和说话人验证，保留 MFCC、FBank 对照。没有加入跨域适应，也没有加入 Res2Net 或 X-Vector。代码可重新训练，但当前没有完整研究数据和已确认来源的历史 checkpoint，不能据此保证复现论文中的某个数值。

## 五个主要目录

| 文件夹 | 主要文件 | 功能 |
|---|---|---|
| Dataset | data.py、prepare.py、demo.py | 读取音频、准备训练清单、生成合成演示音频 |
| Features | spectrum.py、afsc.py、frontend.py、export.py | 功率谱与三种声学特征、导出学习后的滤波器 |
| Model | ecapa_tdnn.py、system.py、train.py | ECAPA、分类损失、两阶段训练与恢复 |
| Evaluation | metrics.py、score.py、test_pipeline.py、smoke_test.py | 评分、EER/MinDCF、验证代码 |
| Inference | extract.py、verify.py | 提取嵌入、比较两段音频 |

训练代码放在 Model 中，因此不额外增加第六个 Training 目录。所有命令都从项目根目录运行。

## 安装

使用 Python 3.10–3.12。先安装相匹配的 PyTorch 与 torchaudio 2.5.1，再安装 requirements.txt。CPU 版本示例：

```bash
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## 先跑通演示

```bash
python -m Dataset.demo --output demo_data
python -m Model.train --config Model/demo.yaml --train-csv demo_data/train.csv --output runs/demo --device cpu
python -m Inference.extract --checkpoint runs/demo/last.pt --csv demo_data/eval.csv --output runs/demo/embeddings.npz --device cpu
python -m Evaluation.score --enrol runs/demo/embeddings.npz --trials demo_data/trials.txt --output runs/demo/scores
```

演示使用人工合成的音调和小型 ECAPA，检查程序是否能运行。它不是实际语音，不用于支持论文结论。

## 换成实际数据

准备 16 kHz 音频，以及训练集的 wav.scp、utt2spk。wav.scp 每行是“音频 ID 路径”，utt2spk 每行是“音频 ID 说话人 ID”。音频文件不复制进仓库。

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

trials 每行是“注册音频 ID 测试音频 ID 标签”，同一人为 1，不同人为 0。请使用与实际数据集协议相符的划分和 trials。代码不会自行猜测 CN-Celeb 的注册集聚合方式，也不会生成替代官方协议的随机研究测试集。

在训练命令中增加 `--feature fbank` 或 `--feature mfcc` 即可切换基线，同时更换输出目录。提取时会自动从 checkpoint 读取特征类型，不需要重新指定。

## 配置与原有代码的区别

- `Model/config.yaml`：公开版默认，联合训练 2 轮、冻结训练 8 轮，margin 固定为 0.3。
- `Model/legacy_schedule.yaml`：保留日志对应的训练时序选项，2+10 轮、margin 从 0 增长到 0.3，学习率在第 10 轮末达到最小值。它只能表达训练日程，不能证明对应了某个论文结果。
- `Model/demo.yaml`：缩小模型、合成音频验证专用。
- AFSC 保留原始代码的初始化参考数组、正间隔归一化和滤波公式。其精确边界约为 20.039–7614.844 Hz，首次实际输出的点也会受归一化影响。没有悄悄将它替换成新的“精确 20–7600 Hz 初始化”。
- 新 CSV 默认每条音频一行，在训练时随机裁剪 3 秒。用户主动提供 start/stop 时，会先按该区间读取，再裁剪。原代码“CSV 多行分段但训练忽略区间”的行为没有继续沿用。
- 新训练器保存完整状态以恢复训练，采用新的 checkpoint 格式。旧的 embedding_model.ckpt 不能直接加载。
- 评分显式处理相同分数和 ROC 端点，所以在分数相同的情况下可能与旧指标函数有细微差别。

详细参数、格式与归属说明见英文 README.md。

## 训练恢复及查看结果

```bash
python -m Model.train --config Model/config.yaml --train-csv /data/train/train.csv --output runs/afsc --resume runs/afsc/last.pt --device cuda
python -m Features.export --checkpoint runs/afsc/last.pt --output runs/afsc/filters --plot
```

训练输出包括逐轮日志、配置、说话人编号、完整 checkpoint 和频率点。恢复到下一轮，不支持轮中间恢复。完整 checkpoint 内含数据清单路径，准备公开模型时应检查这些路径。

评分输出 `scores.csv` 和 `metrics.json`。EER 的单位是百分比，MinDCF 使用归一化值，默认目标先验为 0.01。

## 检查程序

```bash
python -m unittest Evaluation.test_pipeline -v
python -m Evaluation.smoke_test
```

当前测试范围见 TESTING.md。测试通过说明软件流程可运行，不能代替真实数据集训练和实验验证。
