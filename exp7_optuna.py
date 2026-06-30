import os
import random

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_curve,precision_score,recall_score,auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


seed=42
data_path="data/creditcard.csv"
out_dir="results_test_only"
n_trials=int(os.environ.get("OPTUNA_TRIALS","12"))
thresholds=[0.01,0.03,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90]


def set_seed():
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    df=pd.read_csv(data_path)
    X=df.drop(columns=["Class"])
    y=df["Class"].astype(int)

    X_train_val,X_test,y_train_val,y_test=train_test_split(
        X,y,test_size=0.3,random_state=seed,stratify=y
    )
    X_train,X_val,y_train,y_val=train_test_split(
        X_train_val,y_train_val,test_size=0.2,random_state=seed,stratify=y_train_val
    )

    for data in [X_train,X_val,X_test]:
        data.reset_index(drop=True,inplace=True)
    y_train=y_train.reset_index(drop=True)
    y_val=y_val.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)

    scaler=StandardScaler()
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
    rows=[]
    for name in files:
        path=os.path.join(out_dir,name)
        if os.path.exists(path):
            rows.append(pd.read_csv(path))
    pd.concat(rows,ignore_index=True,sort=False).to_csv(
        os.path.join(out_dir,"all_experiments_summary.csv"),
        index=False,encoding="utf-8-sig"
    )


def main():
    set_seed()
    os.makedirs(out_dir,exist_ok=True)
    X_train,X_val,X_test,y_train,y_val,y_test=load_data()

    print("\n"+"="*70)
    print("Training Random Forest with SMOTE and Optuna")
    print("="*70)

    neg_count=(y_train==0).sum()
    pos_count=(y_train==1).sum()
    print("train normal:",neg_count)
    print("train fraud:",pos_count)

    def rf_smote_objective(trial):
        sampling_strategy=trial.suggest_categorical("sampling_strategy",[0.1,0.3,0.5,1.0])
        k_neighbors=trial.suggest_categorical("k_neighbors",[3,5,7])

        smote=SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=seed
        )
        X_res,y_res=smote.fit_resample(X_train,y_train)

        params={
            "n_estimators":25,
            "max_depth":10,
            "min_samples_split":2,
            "min_samples_leaf":2,
            "max_features":"sqrt",
            "random_state":seed,
            "n_jobs":-1
        }

        model=RandomForestClassifier(**params)
        model.fit(X_res,y_res)
        prob=model.predict_proba(X_val)[:,1]
        precision,recall,_=precision_recall_curve(y_val,prob)
        return auc(recall,precision)

    sampler=optuna.samplers.TPESampler(seed=seed)
    study=optuna.create_study(direction="maximize",sampler=sampler)
    study.enqueue_trial({
        "sampling_strategy":0.10,
        "k_neighbors":5,
    })
    study.optimize(rf_smote_objective,n_trials=n_trials,show_progress_bar=False)

    print("Optuna best validation AUPRC:",round(study.best_value,6))

    best=study.best_params
    smote=SMOTE(
        sampling_strategy=best["sampling_strategy"],
        k_neighbors=best["k_neighbors"],
        random_state=seed
    )
    X_res,y_res=smote.fit_resample(X_train,y_train)

    model=RandomForestClassifier(
        n_estimators=25,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_res,y_res)
    test_prob=model.predict_proba(X_test)[:,1]
    test_metrics=calc_metrics(y_test,test_prob,0.5)

    summary={
        "experiment_id":"exp7",
        "experiment_name":"optuna_rf_smote",
        "split":"test",
        "best_trial_number":study.best_trial.number,
        "model":"RF",
        "method":"SMOTE",
        "sampling_strategy":best["sampling_strategy"],
        "k_neighbors":best["k_neighbors"],
        "n_estimators":25,
        "max_depth":10,
        "min_samples_split":2,
        "min_samples_leaf":2,
        "max_features":"sqrt",
        "test_AUPRC":test_metrics["AUPRC"],
        "test_Average_Precision":test_metrics["Average_Precision"],
        "test_Custom_Score":test_metrics["Custom_Score"]
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir,"exp7_optuna_search_summary.csv"),
        index=False,encoding="utf-8-sig"
    )

    scan_rows=[]
    for threshold in thresholds:
        row={
            "experiment_id":"exp7",
            "experiment_name":"best_model_threshold_scan",
            "model":"RF",
            "training_method":"SMOTE",
            "split":"test",
            "sampling_strategy":best["sampling_strategy"],
            "k_neighbors":best["k_neighbors"]
        }
        row.update(calc_metrics(y_test,test_prob,threshold))
        scan_rows.append(row)
    pd.DataFrame(scan_rows).to_csv(
        os.path.join(out_dir,"exp7_best_model_threshold_scan.csv"),
        index=False,encoding="utf-8-sig"
    )

    write_summary_file()
    print("Test AUPRC:",round(test_metrics["AUPRC"],6))


if __name__=="__main__":
    main()
