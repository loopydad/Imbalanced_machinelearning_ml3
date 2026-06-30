import os
import random

import numpy as np
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
THRESHOLDS=[0.05,0.10,0.20,0.30,0.50,0.70]


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


def build_model(name,method,weight_text):
    cw=None
    if method=="class_weight" and name in ["LR","RF"]:
        cw="balanced" if weight_text=="balanced" else {0:1,1:float(weight_text)}
    if name=="LR":
        return LogisticRegression(max_iter=1000,class_weight=cw,random_state=SEED)
    if name=="RF":
        return RandomForestClassifier(n_estimators=25,max_depth=10,min_samples_leaf=2,class_weight=cw,n_jobs=-1,random_state=SEED)
    return MLPClassifier(hidden_layer_sizes=(16,),max_iter=15,batch_size=2048,early_stopping=True,random_state=SEED)


def make_train(X_train,y_train,method,strategy,k):
    if method=="SMOTE":
        smote=SMOTE(sampling_strategy=strategy,k_neighbors=int(k),random_state=SEED)
        return smote.fit_resample(X_train,y_train)
    if method=="SMOTEENN":
        smote=SMOTE(sampling_strategy=strategy,k_neighbors=int(k),random_state=SEED)
        sampler=SMOTEENN(smote=smote,random_state=SEED,n_jobs=-1)
        return sampler.fit_resample(X_train,y_train)
    return X_train,y_train


def make_weight(y,weight_text):
    if weight_text=="balanced":
        return compute_sample_weight(class_weight="balanced",y=y)
    return np.where(np.asarray(y)==1,float(weight_text),1.0)


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


def pick_best_smote(model_name,X_train,X_val,y_train,y_val):
    best_score=-1
    best_config=None
    for strategy in [0.1,0.3,0.5,1.0]:
        for k in [3,5,7]:
            smote=SMOTE(sampling_strategy=strategy,k_neighbors=k,random_state=SEED)
            X_fit,y_fit=smote.fit_resample(X_train,y_train)
            model=build_model(model_name,"SMOTE",None)
            model.fit(X_fit,y_fit)
            prob=model.predict_proba(X_val)[:,1]
            score=calc_metrics(y_val,prob,0.5)["AUPRC"]
            if score>best_score:
                best_score=score
                best_config={"method":"SMOTE","strategy":strategy,"k":k,"weight":None}
    return best_config


def pick_best_weight(model_name,X_train,X_val,y_train,y_val):
    best_score=-1
    best_config=None
    for weight_text in ["1","2","5","10","20","balanced"]:
        model=build_model(model_name,"class_weight",weight_text)
        if model_name=="MLP":
            model.fit(X_train,y_train,sample_weight=make_weight(y_train,weight_text))
        else:
            model.fit(X_train,y_train)
        prob=model.predict_proba(X_val)[:,1]
        score=calc_metrics(y_val,prob,0.5)["AUPRC"]
        if score>best_score:
            best_score=score
            best_config={"method":"class_weight","strategy":None,"k":None,"weight":weight_text}
    return best_config


def main():
    set_seed()
    os.makedirs(OUT_DIR,exist_ok=True)
    X_train,X_val,X_test,y_train,y_val,y_test=load_split()
    selected=[]
    rows=[]

    for model_name in ["LR","RF","MLP"]:
        configs=[
            ("none",{"method":"none","strategy":None,"k":None,"weight":None}),
            ("best_SMOTE",pick_best_smote(model_name,X_train,X_val,y_train,y_val)),
            ("best_class_weight",pick_best_weight(model_name,X_train,X_val,y_train,y_val)),
            ("SMOTEENN_default",{"method":"SMOTEENN","strategy":1.0,"k":5,"weight":None})
        ]
        for config_name,config in configs:
            if config is None:
                continue
            method=config["method"]
            X_fit,y_fit=make_train(X_train,y_train,method,config["strategy"],config["k"])
            model=build_model(model_name,method,config["weight"])
            if model_name=="MLP" and method=="class_weight":
                model.fit(X_fit,y_fit,sample_weight=make_weight(y_fit,config["weight"]))
            else:
                model.fit(X_fit,y_fit)

            selected.append({
                "model":model_name,
                "selected_config_type":config_name,
                "selected_training_method":method,
                "selected_sampling_strategy":config["strategy"],
                "selected_k_neighbors":config["k"],
                "selected_class_weight":config["weight"]
            })

            test_prob=model.predict_proba(X_test)[:,1]
            for threshold in THRESHOLDS:
                row={
                    "experiment_id":"exp6","experiment_name":"selected_method_threshold",
                    "model":model_name,"training_method":method,"selected_config_type":config_name,
                    "split":"test","sampling_strategy":config["strategy"],"k_neighbors":config["k"],
                    "class_weight_setting":config["weight"]
                }
                row.update(calc_metrics(y_test,test_prob,threshold))
                rows.append(row)

    pd.DataFrame(selected).to_csv(os.path.join(OUT_DIR,"exp6_selected_training_configs.csv"),index=False,encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"exp6_selected_training_methods_threshold_scan.csv"),index=False,encoding="utf-8-sig")
    print("exp6 done")


if __name__=="__main__":
    main()
