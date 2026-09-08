# 信用卡欺诈检测中的不平衡学习

> Comparing imbalance-learning strategies for credit card fraud detection

本项目研究信用卡欺诈检测中的**类别不平衡问题**：欺诈交易只占全部交易的极少部分，仅使用 Accuracy（准确率）容易得到看似很高、实际却漏掉大量欺诈交易的模型。

项目基于公开的 Credit Card Fraud Detection 数据集，对比 Logistic Regression、Random Forest 和 MLP 三类模型，并研究 Class Weight、SMOTE、SMOTEENN 与分类阈值调整对欺诈识别效果的影响。

## 项目亮点

- 使用分层抽样划分训练集、验证集和测试集，保持各数据集中的欺诈比例基本一致。
- 标准化和重采样均在划分数据后进行，避免将测试集信息用于模型训练。
- 不以 Accuracy 作为主要指标，而是综合比较 AUPRC、Fraud-F1、Precision、Recall 和混淆矩阵。
- 对 SMOTE 采样比例、近邻数和类别权重进行消融实验，分析不同参数对误报与漏报的影响。
- 研究分类阈值变化带来的业务权衡：降低阈值通常可以减少漏报，但会增加正常交易误报。

## 数据集

项目使用 Kaggle 的 [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 数据集。

- 原始数据共 284,807 条交易，其中 492 条为欺诈交易。
- `V1` 至 `V28` 是经过 PCA 匿名化处理的特征。
- `Time` 和 `Amount` 分别表示交易时间与金额。
- `Class=1` 表示欺诈交易，`Class=0` 表示正常交易。

本项目按照约 `56% / 14% / 30%` 的比例划分训练集、验证集和测试集。测试集包含 85,443 条交易，其中 148 条为欺诈交易。

由于数据文件较大且受数据源许可约束，仓库不直接包含原始数据。请下载 `creditcard.csv` 并放入以下位置：

```text
data/
└── creditcard.csv
```

## 实验设计

| 实验 | 内容 | 目的 |
| --- | --- | --- |
| Exp 1 | LR、RF、MLP 基线比较 | 比较不同模型在原始不平衡数据上的表现 |
| Exp 2 | None、Class Weight、SMOTE、SMOTEENN | 比较常见不平衡处理方法 |
| Exp 3 | SMOTE 参数消融 | 分析采样比例和近邻数的影响 |
| Exp 4 | Class Weight 参数消融 | 分析少数类权重变化的影响 |
| Exp 5 | 分类阈值扫描 | 观察 Precision、Recall、FP 和 FN 的变化 |
| Exp 6 | 代表性训练方法与阈值组合 | 分析训练期处理与阈值调整的组合效果 |

### 评价指标

- **AUPRC**：衡量模型能否把少数类欺诈交易排在前面，适合类别严重不平衡的任务。
- **Fraud-F1**：综合衡量欺诈类别的 Precision 和 Recall。
- **Precision**：模型判定为欺诈的交易中，实际欺诈所占比例。
- **Recall**：全部真实欺诈交易中，被模型成功识别的比例。
- **FP / FN**：分别表示正常交易被误报为欺诈、欺诈交易被漏判为正常的数量。
- **Custom Score**：用于观察不同误报和漏报成本假设下的模型表现，定义为 `10 × TP - 20 × FN - FP`。该分数是实验性业务指标，不代表真实金融损失。

## 主要结果

### 1. 基线模型

在默认分类阈值 `0.5` 下，Random Forest 的综合表现最好：

| 模型 | AUPRC | Fraud-F1 | Precision | Recall | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7078 | 0.7315 | 0.8624 | 0.6351 | 15 | 54 |
| Random Forest | **0.8051** | **0.8203** | **0.9722** | **0.7095** | **3** | **43** |
| MLP | 0.5999 | 0.7037 | 0.7787 | 0.6419 | 27 | 53 |

### 2. SMOTE 参数选择

在验证集上选择 SMOTE 配置后，Random Forest 使用 `sampling_strategy=0.1`、`k_neighbors=5`，在留出测试集上的 AUPRC 达到 **0.8341**，相比未进行不平衡处理的 Random Forest（0.8051）提高约 **0.029**。

结果也表明，更强的过采样并不一定更好。将少数类直接补到与多数类相同数量，可能明显增加误报。

### 3. 阈值敏感性分析

对于未重采样的 Random Forest，将阈值从 `0.5` 降低至 `0.2` 时：

| 阈值 | Precision | Recall | FP | FN | Custom Score |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.9722 | 0.7095 | 3 | 43 | 187 |
| 0.2 | 0.8133 | 0.8243 | 28 | 26 | 672 |

降低阈值使欺诈召回率提高约 **11.5 个百分点**，同时误报数由 3 增加到 28。这说明欺诈检测不能只追求单一算法指标，还需要根据实际误报和漏报成本选择阈值。

> 注意：当前阈值结果用于测试集上的敏感性分析，不应被视为独立测试集上无偏估计的“最优阈值”。在真实部署或更严格的实验中，应仅在验证集上选择阈值，再对测试集进行一次最终评估。

更完整的结果见 [`report.pdf`](report.pdf)、[`report_tables/`](report_tables/) 和 [`results_test_only/`](results_test_only/)。

## 快速开始

### 1. 获取代码

```bash
git clone https://github.com/loopydad/Imbalanced_machinelearning_ml3.git
cd Imbalanced_machinelearning_ml3
```

### 2. 创建环境并安装依赖

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell 激活虚拟环境的命令为：

```powershell
.venv\Scripts\Activate.ps1
```

### 3. 准备数据

从 Kaggle 下载数据集，将文件保存为：

```text
data/creditcard.csv
```

### 4. 运行实验

运行全部实验：

```bash
sh run.sh
```

也可以单独运行某个实验，例如：

```bash
python exp1_baseline.py
python exp3_smote.py
python exp6_selected_threshold.py
```

实验结果会写入 `results_test_only/`。

## 项目结构

```text
.
├── exp1_baseline.py                 # 基线模型比较
├── exp2_train_methods.py            # 不平衡处理方法比较
├── exp3_smote.py                    # SMOTE 参数消融
├── exp4_class_weight.py             # 类别权重消融
├── exp5_threshold.py                # 阈值敏感性分析
├── exp6_selected_threshold.py       # 代表性方法与阈值组合实验
├── report_figures/                  # 实验图表
├── report_tables/                   # 报告汇总表格
├── results_test_only/               # 完整实验结果
├── report.pdf                       # 项目报告
├── requirements.txt                 # Python 依赖
└── run.sh                           # 全部实验运行入口
```

## 结论

1. 在极度不平衡的数据上，Accuracy 无法充分反映模型识别欺诈交易的能力。
2. 本实验中 Random Forest 的基线表现优于 Logistic Regression 和当前配置下的 MLP。
3. Class Weight、SMOTE 和 SMOTEENN 通常能提高 Recall，但也可能引入大量误报。
4. 不平衡处理方法对参数较为敏感；较强的重采样或权重设置不一定带来更好的综合表现。
5. 分类阈值是控制误报与漏报的重要手段，应结合实际业务成本在验证集上选择。

## 局限与后续工作

- 将阈值选择完全移至验证集，并保留测试集用于最终一次评估。
- 使用多次分层交叉验证，报告均值和标准差，降低单次数据划分带来的偶然性。
- 增加概率校准，并研究不同业务成本假设下的决策阈值。
- 补充 XGBoost、LightGBM 等常用于表格数据的模型，并进行统一的超参数搜索。
- 将重复的数据处理和指标计算逻辑抽取为公共模块，提高代码可维护性。

## 技术栈

Python · Pandas · NumPy · Scikit-learn · Imbalanced-learn · Matplotlib · Seaborn

## 项目成员

葛奕、邱皓哲、徐鹤岩

