import os
import random

import numpy as np
import optuna
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_curve,precision_score,recall_score,auc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


SEED=42
DATA_PATH="data/creditcard.csv"
OUT_DIR="simple_optuna_experiments/results_test_only"
N_TRIALS=int(os.environ.get("OPTUNA_TRIALS","30"))
THRESHOLDS=[0.01,0.03,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)


def load_split():
    df=pd.read_csv(DATA_PATH)
    X=df.drop(columns=["Class"])
    y=df["Class"].astype(int)
    X_train_val,X_test,y_train_val,y_test=train_test_split(X,y,test_size=0.3,random_state=SEED,stratify=y)
    X_train,X_val,y_train,y_val=train_test_split(X_train_val,y_train_val,test_size=0.2,random_state=SEED,stratify=y_train_val)
    scaler=StandardScaler()
    for data in [X_train,X_val,X_test]:
        data.reset_index(drop=True,inplace=True)
    y_train=y_train.reset_index(drop=True)
    y_val=y_val.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)
    cols=["Time","Amount"]
    X_train[cols]=scaler.fit_transform(X_train[cols])
    X_val[cols]=scaler.transform(X_val[cols])
    X_test[cols]=scaler.transform(X_test[cols])
    return X_train,X_val,X_test,y_train,y_val,y_test


def calc_metrics(y_true,prob,threshold):
    pred=(prob>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_true,pred,labels=[0,1]).ravel()
    precision,recall,_=precision_recall_curve(y_true,prob)
    return {
        "Accuracy":accuracy_score(y_true,pred),
        "Balanced_Accuracy":balanced_accuracy_score(y_true,pred),
        "Macro_F1":f1_score(y_true,pred,average="macro",zero_division=0),
        "Fraud_F1":f1_score(y_true,pred,pos_label=1,zero_division=0),
        "Precision":precision_score(y_true,pred,zero_division=0),
        "Recall":recall_score(y_true,pred,zero_division=0),
        "AUPRC":auc(recall,precision),
        "Average_Precision":average_precision_score(y_true,prob),
        "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),
        "threshold":threshold,
        "Custom_Score":int(tp*10-fn*20-fp)
    }


def make_train(X_train,y_train,method,strategy,k):
    if method=="SMOTE":
        smote=SMOTE(sampling_strategy=strategy,k_neighbors=k,random_state=SEED)
        return smote.fit_resample(X_train,y_train)
    if method=="SMOTEENN":
        smote=SMOTE(sampling_strategy=strategy,k_neighbors=k,random_state=SEED)
        sampler=SMOTEENN(smote=smote,random_state=SEED,n_jobs=-1)
        return sampler.fit_resample(X_train,y_train)
    return X_train,y_train


def make_model(params):
    name=params["model"]
    method=params["method"]
    cw=None
    if method=="class_weight" and name in ["LR","RF"]:
        cw="balanced" if params["weight_mode"]=="balanced" else {0:1,1:params["pos_weight"]}

    if name=="LR":
        return LogisticRegression(C=params["C"],class_weight=cw,max_iter=1000,random_state=SEED)
    if name=="RF":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=cw,
            n_jobs=-1,
            random_state=SEED
        )
    return MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        max_iter=params["max_iter"],
        batch_size=2048,
        early_stopping=True,
        random_state=SEED
    )


def make_mlp_weight(y,params):
    if params["method"]!="class_weight" or params["model"]!="MLP":
        return None
    if params["weight_mode"]=="balanced":
        return compute_sample_weight(class_weight="balanced",y=y)
    return np.where(np.asarray(y)==1,params["pos_weight"],1.0)


def ask_params(trial):
    params={}
    params["model"]=trial.suggest_categorical("model",["LR","RF","MLP"])
    params["method"]=trial.suggest_categorical("method",["none","class_weight","SMOTE","SMOTEENN"])

    params["C"]=trial.suggest_float("C",0.01,10.0,log=True)
    params["n_estimators"]=trial.suggest_int("n_estimators",20,80)
    params["max_depth"]=trial.suggest_int("max_depth",4,14)
    params["min_samples_split"]=trial.suggest_int("min_samples_split",2,10)
    params["min_samples_leaf"]=trial.suggest_int("min_samples_leaf",1,5)
    params["max_features"]=trial.suggest_categorical("max_features",["sqrt","log2",None])
    params["hidden_code"]=trial.suggest_categorical("hidden_code",["16","32","32_16"])
    if params["hidden_code"]=="16":
        params["hidden_layer_sizes"]=(16,)
    elif params["hidden_code"]=="32":
        params["hidden_layer_sizes"]=(32,)
    else:
        params["hidden_layer_sizes"]=(32,16)
    params["alpha"]=trial.suggest_float("alpha",1e-5,1e-2,log=True)
    params["learning_rate_init"]=trial.suggest_float("learning_rate_init",1e-4,3e-3,log=True)
    params["max_iter"]=trial.suggest_int("max_iter",10,30)

    params["weight_mode"]=trial.suggest_categorical("weight_mode",["balanced","manual"])
    params["pos_weight"]=trial.suggest_float("pos_weight",1.0,30.0)
    params["sampling_strategy"]=trial.suggest_float("sampling_strategy",0.05,1.0)
    params["k_neighbors"]=trial.suggest_int("k_neighbors",3,10)
    return params


def train_and_score(params,X_train,y_train,X_eval,y_eval):
    X_fit,y_fit=make_train(X_train,y_train,params["method"],params["sampling_strategy"],params["k_neighbors"])
    model=make_model(params)
    weight=make_mlp_weight(y_fit,params)
    if weight is None:
        model.fit(X_fit,y_fit)
    else:
        model.fit(X_fit,y_fit,sample_weight=weight)
    prob=model.predict_proba(X_eval)[:,1]
    score=calc_metrics(y_eval,prob,0.5)["AUPRC"]
    return score,model,prob


def params_to_row(params):
    row=params.copy()
    if row["method"]!="class_weight":
        row["weight_mode"]=None
        row["pos_weight"]=None
    if row["method"] not in ["SMOTE","SMOTEENN"]:
        row["sampling_strategy"]=None
        row["k_neighbors"]=None
    return row


def write_summary_file():
    files=[
        "exp1_baseline_model_comparison.csv",
        "exp2_imbalance_method_comparison.csv",
        "exp3_smote_ablation.csv",
        "exp4_class_weight_ablation.csv",
        "exp5_threshold_moving_ablation.csv",
        "exp6_selected_training_methods_threshold_scan.csv",
        "exp7_optuna_search_summary.csv",
        "exp7_best_model_threshold_scan.csv"
    ]
    dfs=[]
    for name in files:
        path=os.path.join(OUT_DIR,name)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if len(dfs)>0:
        pd.concat(dfs,ignore_index=True,sort=False).to_csv(os.path.join(OUT_DIR,"all_experiments_summary.csv"),index=False,encoding="utf-8-sig")

    text=(
        "# 实验说明\n\n"
        "7个脚本分别对应7个实验。所有脚本都固定随机种子，先划分train/val/test，再标准化Time和Amount。\n\n"
        "AUPRC和Average Precision看的是概率排序能力，不依赖某个固定threshold。\n\n"
        "Custom Score=TP*10-FN*20-FP，漏掉欺诈的惩罚最大。\n\n"
        "实验7用Optuna搜索模型、训练方法和参数，目标是validation AUPRC。选出最好trial后，只在test上评估一次。\n"
    )
    with open(os.path.join(OUT_DIR,"experiment_readme.md"),"w",encoding="utf-8") as f:
        f.write(text)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    set_seed()
    os.makedirs(OUT_DIR,exist_ok=True)
    X_train,X_val,X_test,y_train,y_val,y_test=load_split()
    trial_rows=[]

    def objective(trial):
        params=ask_params(trial)
        row={"trial_number":trial.number}
        row.update(params_to_row(params))
        try:
            val_auprc,model,val_prob=train_and_score(params,X_train,y_train,X_val,y_val)
            val_metrics=calc_metrics(y_val,val_prob,0.5)
            row.update({"validation_AUPRC":val_metrics["AUPRC"],"validation_Average_Precision":val_metrics["Average_Precision"],"validation_Custom_Score":val_metrics["Custom_Score"]})
            score=val_auprc
        except Exception as e:
            print("trial",trial.number,e)
            row["validation_AUPRC"]=0.0
            score=0.0
        trial_rows.append(row)
        return score

    sampler=optuna.samplers.TPESampler(seed=SEED)
    study=optuna.create_study(direction="maximize",sampler=sampler)
    study.optimize(objective,n_trials=N_TRIALS)

    best_params=ask_params(study.best_trial)
    test_auprc,best_model,test_prob=train_and_score(best_params,X_train,y_train,X_test,y_test)
    test_metrics=calc_metrics(y_test,test_prob,0.5)

    summary={"experiment_id":"exp7","experiment_name":"optuna_search","split":"test","selection_metric":"validation_AUPRC","best_trial_number":study.best_trial.number}
    summary.update(params_to_row(best_params))
    summary.update({"test_AUPRC":test_metrics["AUPRC"],"test_Average_Precision":test_metrics["Average_Precision"],"test_Custom_Score":test_metrics["Custom_Score"]})
    pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR,"exp7_optuna_search_summary.csv"),index=False,encoding="utf-8-sig")

    scan_rows=[]
    for threshold in THRESHOLDS:
        row={"experiment_id":"exp7","experiment_name":"best_model_threshold_scan","model":best_params["model"],"training_method":best_params["method"],"split":"test"}
        row.update(params_to_row(best_params))
        row.update(calc_metrics(y_test,test_prob,threshold))
        scan_rows.append(row)
    pd.DataFrame(scan_rows).to_csv(os.path.join(OUT_DIR,"exp7_best_model_threshold_scan.csv"),index=False,encoding="utf-8-sig")

    write_summary_file()
    print("exp7 done")
    print("best validation AUPRC:",study.best_value)
    print("test AUPRC:",test_metrics["AUPRC"])


if __name__=="__main__":
    main()
