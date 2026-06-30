# 结果说明

这个目录只放最终查看用的结果。

所有表都删掉了 status 和 skipped_reason。

有 split 字段的表只保留 test。验证集只在代码内部用于选参数，不在最终结果里单独展示。

Optuna 的最终表使用 test_AUPRC、test_Average_Precision 和 test_Custom_Score 汇报最优模型在测试集上的表现。
