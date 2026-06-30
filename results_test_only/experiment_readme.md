# 实验说明

7个脚本分别对应7个实验。所有脚本都固定随机种子，先划分train/val/test，再标准化Time和Amount。

AUPRC和Average Precision看的是概率排序能力，不依赖某个固定threshold。

Custom Score=TP*10-FN*20-FP，漏掉欺诈的惩罚最大。

实验7用Optuna搜索模型、训练方法和参数。参数选择只看验证集AUPRC，最终表只汇报测试集结果。
