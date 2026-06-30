import os

import pandas as pd


base="results_test_only"
out="report_tables"
os.makedirs(out,exist_ok=True)

metric_cols=["AUPRC","Average_Precision","Fraud_F1","Precision","Recall","Macro_F1","Balanced_Accuracy"]


def read(name):
    return pd.read_csv(os.path.join(base,name))


def tidy(df):
    df=df.copy()
    df=df.replace({pd.NA:"", float("nan"):""})
    for col in metric_cols:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col],errors="coerce").round(4)
    for col in ["threshold","sampling_strategy"]:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col],errors="coerce").round(4)
    for col in ["FP","FN","TP","TN","Custom_Score","k_neighbors"]:
        if col in df.columns:
            df[col]=pd.to_numeric(df[col],errors="coerce").astype("Int64")
    return df


def save_table(df,name):
    df=tidy(df)
    df.to_csv(os.path.join(out,name+".csv"),index=False,encoding="utf-8-sig")
    return df


def class_distribution():
    candidates=[
        os.path.join("data","creditcard.csv"),
        os.path.join("..","data","creditcard.csv")
    ]
    data_path=None
    for path in candidates:
        if os.path.exists(path):
            data_path=path
            break

    if data_path is None:
        return pd.DataFrame({
            "Class":[0,1],
            "Meaning":["Normal","Fraud"],
            "Count":["not uploaded","not uploaded"],
            "Ratio":["",""]
        })

    df=pd.read_csv(data_path,usecols=["Class"])
    count=df["Class"].value_counts().sort_index()
    ratio=df["Class"].value_counts(normalize=True).sort_index()
    return pd.DataFrame({
        "Class":[0,1],
        "Meaning":["Normal","Fraud"],
        "Count":[int(count.get(0,0)),int(count.get(1,0))],
        "Ratio":[round(float(ratio.get(0,0)),6),round(float(ratio.get(1,0)),6)]
    })


table1=save_table(class_distribution(),"table1_class_distribution")

exp1=read("exp1_baseline_model_comparison.csv")
table2=save_table(
    exp1[["model","AUPRC","Average_Precision","Fraud_F1","Precision","Recall","FP","FN","TP","Custom_Score"]],
    "table2_baseline"
)

exp2=read("exp2_imbalance_method_comparison.csv")
table3=save_table(
    exp2[["model","training_method","AUPRC","Average_Precision","Fraud_F1","Precision","Recall","FP","FN","TP","Custom_Score"]],
    "table3_imbalance_methods"
)

exp3=read("exp3_smote_ablation.csv")
table4_raw=exp3.sort_values(["model","AUPRC"],ascending=[True,False]).groupby("model").head(3)
table4=save_table(
    table4_raw[["model","sampling_strategy","k_neighbors","AUPRC","Average_Precision","Fraud_F1","Precision","Recall","FP","FN","TP","Custom_Score"]],
    "table4_smote_top_configs"
)

exp4=read("exp4_class_weight_ablation.csv")
table5=save_table(
    exp4[["model","class_weight_setting","AUPRC","Average_Precision","Fraud_F1","Precision","Recall","FP","FN","TP","Custom_Score"]],
    "table5_class_weight_ablation"
)

exp5=read("exp5_threshold_moving_ablation.csv")
table6_raw=exp5[(exp5["model"]=="RF") & (exp5["threshold"].isin([0.05,0.10,0.20,0.30,0.50,0.70]))]
table6=save_table(
    table6_raw[["model","threshold","AUPRC","Average_Precision","Fraud_F1","Precision","Recall","FP","FN","TP","Custom_Score"]],
    "table6_rf_threshold_scan"
)

exp6=read("exp6_selected_training_methods_threshold_scan.csv")
table7_raw=exp6[(exp6["model"]=="RF") & (exp6["threshold"].isin([0.10,0.20,0.30,0.50,0.70]))]
table7=save_table(
    table7_raw[["model","selected_config_type","training_method","threshold","sampling_strategy","k_neighbors","class_weight_setting","AUPRC","Average_Precision","Fraud_F1","Precision","Recall","FP","FN","TP","Custom_Score"]],
    "table7_rf_selected_methods_threshold"
)

exp7=read("exp7_optuna_search_summary.csv")
exp7_scan=read("exp7_best_model_threshold_scan.csv")
table8=exp7[["model","method","sampling_strategy","k_neighbors","n_estimators","max_depth","min_samples_leaf","max_features","test_AUPRC","test_Average_Precision","test_Custom_Score"]]
table8=table8.rename(columns={
    "test_AUPRC":"AUPRC",
    "test_Average_Precision":"Average_Precision",
    "test_Custom_Score":"Custom_Score"
})
table8=save_table(table8,"table8_optuna_best")

summary_rows=[]
rf_base=exp1[exp1["model"]=="RF"].iloc[0].copy()
rf_base["Method"]="Baseline RF"
summary_rows.append(rf_base)

rf_smote=exp3[exp3["model"]=="RF"].sort_values("AUPRC",ascending=False).iloc[0].copy()
rf_smote["Method"]="RF + SMOTE best AUPRC"
summary_rows.append(rf_smote)

rf_threshold=exp5[exp5["model"]=="RF"].sort_values("Custom_Score",ascending=False).iloc[0].copy()
rf_threshold["Method"]="RF + threshold moving"
summary_rows.append(rf_threshold)

rf_smote_threshold=exp6[(exp6["model"]=="RF") & (exp6["selected_config_type"]=="best_SMOTE")].sort_values("Custom_Score",ascending=False).iloc[0].copy()
rf_smote_threshold["Method"]="RF + best_SMOTE + threshold"
summary_rows.append(rf_smote_threshold)

optuna=exp7.iloc[0].copy()
optuna_scan=exp7_scan[exp7_scan["threshold"]==0.5].iloc[0].copy()
optuna["Method"]="Optuna RF+SMOTE"
optuna["training_method"]=optuna["method"]
optuna["threshold"]=0.5
optuna["AUPRC"]=optuna["test_AUPRC"]
optuna["Average_Precision"]=optuna["test_Average_Precision"]
optuna["Custom_Score"]=optuna["test_Custom_Score"]
for col in ["Fraud_F1","Precision","Recall","FP","FN","TP"]:
    optuna[col]=optuna_scan[col]
summary_rows.append(optuna)

table9_raw=pd.DataFrame(summary_rows)
table9=save_table(
    table9_raw[["Method","training_method","threshold","sampling_strategy","k_neighbors","AUPRC","Average_Precision","Fraud_F1","Precision","Recall","FP","FN","TP","Custom_Score"]],
    "table9_final_summary"
)

sections=[
    ("表1 数据类别分布",table1),
    ("表2 Baseline 模型比较",table2),
    ("表3 不平衡处理方法比较",table3),
    ("表4 SMOTE 参数消融：各模型 AUPRC 前三组",table4),
    ("表5 Class weight 参数消融",table5),
    ("表6 RF baseline 的 threshold 扫描",table6),
    ("表7 RF 代表性训练方法 + threshold 扫描",table7),
    ("表8 Optuna 最优 RF+SMOTE 配置",table8),
    ("表9 最终重点结果汇总",table9),
]

parts=["# 报告用实验结果表\n"]
for title,df in sections:
    parts.append("## "+title+"\n")
    parts.append(df.to_markdown(index=False))
    parts.append("\n")

with open(os.path.join(out,"report_tables.md"),"w",encoding="utf-8") as f:
    f.write("\n".join(parts))

print("saved to",out)
