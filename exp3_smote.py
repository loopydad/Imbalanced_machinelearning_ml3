import os
import random

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_curve,precision_score,recall_score,auc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


SEED=42
DATA_PATH="data/creditcard.csv"
OUT_DIR="results_test_only"


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


def build_model(name):
    if name=="LR":
        return LogisticRegression(max_iter=1000,random_state=SEED)
    if name=="RF":
        return RandomForestClassifier(n_estimators=25,max_depth=10,min_samples_leaf=2,n_jobs=-1,random_state=SEED)
    return MLPClassifier(hidden_layer_sizes=(16,),max_iter=15,batch_size=2048,early_stopping=True,random_state=SEED)


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


def main():
    set_seed()
    os.makedirs(OUT_DIR,exist_ok=True)
    X_train,X_val,X_test,y_train,y_val,y_test=load_split()
    rows=[]

    for model_name in ["LR","RF","MLP"]:
        for strategy in [0.1,0.3,0.5,1.0]:
            for k in [3,5,7]:
                row_base={
                    "experiment_id":"exp3","experiment_name":"smote_ablation",
                    "model":model_name,"training_method":"SMOTE",
                    "sampling_strategy":strategy,"k_neighbors":k,
                    "class_weight_setting":None
                }
                try:
                    smote=SMOTE(sampling_strategy=strategy,k_neighbors=k,random_state=SEED)
                    X_fit,y_fit=smote.fit_resample(X_train,y_train)
                    model=build_model(model_name)
                    model.fit(X_fit,y_fit)
                    prob=model.predict_proba(X_test)[:,1]
                    row=row_base.copy()
                    row["split"]="test"
                    row.update(calc_metrics(y_test,prob,0.5))
                    rows.append(row)
                except Exception as e:
                    print(model_name,strategy,k,e)
                pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"exp3_smote_ablation.csv"),index=False,encoding="utf-8-sig")

    print("exp3 done")


if __name__=="__main__":
    main()
