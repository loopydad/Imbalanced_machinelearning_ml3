# imbalanced_machinelearning_ml3

信用卡欺诈检测二分类实验。主要比较 baseline、class weight、SMOTE、SMOTEENN、阈值移动和 Optuna 参数搜索。

## 文件说明

- `exp1_baseline.py`：三个模型的 baseline。
- `exp2_train_methods.py`：比较 none、class_weight、SMOTE、SMOTEENN。
- `exp3_smote.py`：SMOTE 参数消融。
- `exp4_class_weight.py`：class weight 参数消融。
- `exp5_threshold.py`：threshold 扫描。
- `exp6_selected_threshold.py`：选出代表性训练配置后再扫 threshold。
- `exp7_optuna.py`：用 Optuna 搜索 RF+SMOTE 的参数。
- `results_test_only/`：已经生成好的 test 结果。
- `data/creditcard.csv`：实验数据，文件较大，不随仓库上传。运行前请自行放到这个路径。

## 运行

Windows:

```bat
run.bat
```

Linux / macOS:

```sh
sh run.sh
```

结果会保存到 `results_test_only/`。

## 说明

原始数据文件较大，仓库中不上传数据，只保留代码和已经生成的 test 结果。
