@echo off
cd /d "%~dp0"
python exp1_baseline.py
python exp2_train_methods.py
python exp3_smote.py
python exp4_class_weight.py
python exp5_threshold.py
python exp6_selected_threshold.py
python exp7_optuna.py
