# Day 1：泰坦尼克号幸存者预测

本项目面向海事博物馆教育团队，用真实 Titanic 乘客记录比较一个简单基线和一个随机森林分类模型，并检查模型在留出数据上的错误。

“分类”是让程序从预先规定的类别中选择答案；本项目预测 `Survived`，其中 `0` 表示历史记录中的未幸存，`1` 表示幸存。

## 小组成员

- 张清波，U202412700（组长）
- 汤家齐，U202412691
- 吕绍康，U202412686

## 项目文件

- `train.py`：数据检查、预处理、训练、评估和错误记录。
- `tests/test_pipeline.py`：预处理、数据划分和候选流水线测试。
- `report.md`：数据来源、方法、结果、失败案例和限制。
- `presentation.pptx`：3 分钟答辩材料。
- `submission.json`：提交清单。
- `team.md`：教师要求的成员名单。

原始数据、运行产生的 `metrics.json` 和 `errors.csv` 不提交到 Git。

## 真实数据

- 所有者/发布者：Kaggle 用户 `hesh97`
- 标题：Titanic-Dataset (train.csv)
- 页面：https://www.kaggle.com/datasets/hesh97/titanicdataset-traincsv
- 许可标签：CC0: Public Domain
- 本组核验日期：2026-08-18
- 预期文件：`data/raw/train.csv`
- 预期规模：891 行、12 列；`Survived` 中 `0` 有 549 行，`1` 有 342 行

只使用上述页面提供的真实 `train.csv`。不要用 AI 生成数据替代下载失败的数据。

## 环境

本次复现通过的环境为：

- Windows PowerShell
- Python 3.12.4
- pandas 3.0.5
- scikit-learn 1.9.0

依赖版本范围写在 `requirements.txt`。其他满足版本范围的环境也应能运行，但结果必须以实际命令输出为准。

## 从零开始运行

以下命令都在仓库根目录运行，也就是能看到 `README.md` 和 `train.py` 的文件夹。

### 1. 取得仓库

```powershell
git clone "https://github.com/boo0517/ai-camp-2026-d1-ZhangQingboU202412700.git"
Set-Location "ai-camp-2026-d1-ZhangQingboU202412700"
```

应该看到：`Get-ChildItem` 能列出 `train.py`、`tests`、`report.md` 和 `presentation.pptx`。

如果不是：运行 `Get-Location`，确认当前文件夹不是课程原仓库或其他日次仓库。

### 2. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

应该看到：pandas 和 scikit-learn 安装成功，命令没有以红色报错结束。

如果出现 `python` 找不到：先运行 `py --version`；若可用，把后续命令中的 `python` 改为 `py`。

### 3. 放置真实数据

从指定 Kaggle 页面下载并解压 `train.csv`，然后在仓库根目录运行：

```powershell
New-Item -ItemType Directory -Force data\raw
Copy-Item "你的下载目录\train.csv" data\raw\train.csv
```

应该看到：`Get-Item data\raw\train.csv` 能找到文件。

如果不是：检查文件名是否仍为 `train.csv`，以及是否多套了一层解压文件夹。仍无法取得时，联系教师获取同一来源的缓存副本。

### 4. 检查数据

```powershell
python train.py --check-data
```

重要输出应包含：

```text
REAL DATA CHECK PASSED
rows: 891
columns: 12
survived_counts: {0: 549, 1: 342}
missing_age: 177
missing_cabin: 687
missing_embarked: 2
```

如果检查失败：停止模型运行，依次检查当前目录、`data/raw/train.csv` 路径、文件名和下载来源，不要修改检查条件来绕过错误。

### 5. 运行测试

```powershell
python -m unittest discover -s tests -v
```

重要输出应包含：

```text
Ran 3 tests
OK
```

如果失败：先处理输出中的第一个失败测试，不要删除测试或降低检查条件。

### 6. 运行模型

```powershell
python train.py
```

应该生成：

- `metrics.json`：基线与候选模型的指标和混淆矩阵。
- `errors.csv`：候选模型在 223 条留出记录上的误判行。

如果提示找不到数据：先重新执行第 4 步；如果没有生成输出文件，保留完整错误并检查依赖安装结果。

## 方法

“基线”是最简单的比较对象。本项目的基线使用 `DummyClassifier(strategy="most_frequent")`，总是预测训练集中最常见的类别。

候选模型使用 `RandomForestClassifier(random_state=42)`。数据按 75%/25% 分层划分为 668 条训练记录和 223 条测试记录，随机种子固定为 42。数值缺失值在流水线中用训练数据的中位数补全；类别缺失值用训练数据的最常见值补全，再进行独热编码。

两个模型使用相同特征、相同划分和相同指标，因此比较条件一致。测试集只用于最后评估，没有参与训练。

## 实际结果

以下结果由 2026-08-18 的完整运行得到：

| 指标 | 多数类基线 | 随机森林候选 |
| --- | ---: | ---: |
| 准确率 | 61.4% | 74.4% |
| 幸存者精确率 | 0.0% | 68.4% |
| 幸存者召回率 | 0.0% | 62.8% |
| 幸存者 F1 | 0.000 | 0.655 |
| 混淆矩阵（标签顺序 0、1） | `[[137, 0], [86, 0]]` | `[[112, 25], [32, 54]]` |

“召回率”表示真实幸存者中被模型识别出的比例。候选模型识别出 54 名留出数据中的真实幸存者，但仍漏掉 32 名；基线漏掉全部 86 名真实幸存者。候选模型还把 25 名实际未幸存者预测为幸存。

## 一个真实失败案例

`errors.csv` 中的 PassengerId 502（`Canavan, Miss. Mary`）实际标签为 `0`，候选模型预测为 `1`。可直接观察到的字段是：三等舱、女性、21 岁、票价 7.75、从 Q 登船。

这个案例证明候选模型会在真实留出记录上出现假阳性。它不能证明这些字段导致了该乘客的历史结局，也不能支持关于救生艇位置或个人经历的故事。

## 使用边界

- 数据来自 1912 年的一次历史事件，不能代表现代人群或现代救援场景。
- 观察数据中的相关性不等于因果关系。
- 当前只比较一个固定随机森林，没有进行多模型或多随机种子稳定性检查。
- 准确率不能单独说明模型是否识别了幸存者，必须与召回率、F1、混淆矩阵和真实错误一起阅读。
- 本项目仅用于历史数据教学展示，不能用于真实救援分配或个人决策。

更完整的数据说明、结果解释和下一项检查见 `report.md`。
