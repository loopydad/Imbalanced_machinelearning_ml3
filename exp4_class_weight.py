import os
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_curve,precision_score,recall_score,auc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


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


def build_model(name,weight_text):
    cw=None
    if name in ["LR","RF"]:
        cw="balanced" if weight_text=="balanced" else {0:1,1:float(weight_text)}
    if name=="LR":
        return LogisticRegression(max_iter=1000,class_weight=cw,random_state=SEED)
    if name=="RF":
        return RandomForestClassifier(n_estimators=25,max_depth=10,min_samples_leaf=2,class_weight=cw,n_jobs=-1,random_state=SEED)
    return MLPClassifier(hidden_layer_sizes=(16,),max_iter=15,batch_size=2048,early_stopping=True,random_state=SEED)


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


def main():
    set_seed()
    os.makedirs(OUT_DIR,exist_ok=True)
    X_train,X_val,X_test,y_train,y_val,y_test=load_split()
    rows=[]

    for model_name in ["LR","RF","MLP"]:
        for weight_text in ["1","2","5","10","20","balanced"]:
            row_base={
                "experiment_id":"exp4","experiment_name":"class_weight_ablation",
                "model":model_name,"training_method":"class_weight",
                "sampling_strategy":None,"k_neighbors":None,
                "class_weight_setting":weight_text
            }
            try:
                model=build_model(model_name,weight_text)
                if model_name=="MLP":
                    model.fit(X_train,y_train,sample_weight=make_weight(y_train,weight_text))
                else:
                    model.fit(X_train,y_train)
                prob=model.predict_proba(X_test)[:,1]
                row=row_base.copy()
                row["split"]="test"
                row.update(calc_metrics(y_test,prob,0.5))
                rows.append(row)
            except Exception as e:
                print(model_name,weight_text,e)
            pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"exp4_class_weight_ablation.csv"),index=False,encoding="utf-8-sig")

    print("exp4 done")


if __name__=="__main__":
    main()
