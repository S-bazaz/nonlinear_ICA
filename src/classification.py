# -*- coding: utf-8 -*-
##############
#  Packages  #
##############
import json
import joblib
import sys
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from torch.utils.data import DataLoader

################
#    Imports   #
################
# # Set the working directory to the parent directory of the script (nonlinear_ICA)
root_path = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, str(root_path))
from src.feature_extraction import get_features_dataset, get_label_from_id_patient, get_features_dataset_parallel

##################
#   Pipelines    #
##################



def gridsearch_SVM_pipeline(X_train,y_train, X_val, y_val, param_grid, scoring='accuracy', verbose=4):
    """Grid search SVM using X_train for training and x_val for valdidation 
    

    Args:
        X_train (np.array): [n_patient_train, n_features]
        y_train (np.array): [n_patient_train]
        X_val (np.array): [n_patient_val, n_features]
        y_val (np.array): [n_patient_val]
        param_grid (dict): dict with keys as SVM parameters name in pipleine, values to test as keys
        scoring (str, optional): Scoring objective in trainin g. Defaults to 'accuracy'.
        verbose (int, optional): Classic verbose parameter. Defaults to 4.
    """
    # Create an SVM pipeline
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])
    split_index = [-1]*len(X_train) + [0]*len(X_val)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    print('in grid search svm pipeline',(X.shape,y.shape))
    pds = PredefinedSplit(test_fold = split_index)
    # Create a GridSearchCV object
    grid_search_model = GridSearchCV(svm_pipeline, param_grid, cv=pds, scoring=scoring, verbose=verbose)
    grid_search_model.fit(X,y)
    results_gs = grid_search_model.cv_results_
    best_params = grid_search_model.best_params_
    best_model = grid_search_model.best_estimator_
    return(results_gs, best_params, best_model)


def get_metrics(y_true, y_pred, y_pred_proba,metrics_to_compute=['accuracy', 'f1_score', 'recall', 'precision']):
    """
    Calculate specified classification metrics.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    - metrics_to_compute: list of metrics to compute
    
    Returns:
    - metrics_dict: a dictionary containing calculated metrics
    """
    valid_metrics = ['accuracy', 'f1_score', 'recall', 'precision', 'roc_auc']
    
    assert all(metric in valid_metrics for metric in metrics_to_compute), "Invalid metric name in metrics_to_compute."

    metrics_dict = {}

    if 'accuracy' in metrics_to_compute:
        metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
    if 'f1_score' in metrics_to_compute:
        metrics_dict['f1_score'] = f1_score(y_true, y_pred)
    if 'recall' in metrics_to_compute:
        metrics_dict['recall'] = recall_score(y_true, y_pred)
    if 'precision' in metrics_to_compute:
        metrics_dict['precision'] = precision_score(y_true, y_pred)
    if 'roc_auc' in metrics_to_compute:
        metrics_dict['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

    return metrics_dict



def get_results(classif_model, X_train, y_train, X_test, y_test, metrics_to_compute=['accuracy', 'f1_score','recall','precision']):
    """Given a pretrained model and train test split, will compute all metrics specified

    Args:
        classif_model (sklearn.model): Pretrained classif model
        X_train (np.array): [n_patient_train, n_features]
        y_train (np.array): [n_patient_train]
        X_test (np.array): [n_patient_test, n_features]
        y_test (np.array): [n_patient_test]
        metrics_to_compute (list, optional): Metrics to compute - chose between accuracy, recall, precision, f1_score, roc_auc. Defaults to ['accuracy', 'f1_score','recall','precision'].

    Returns:
        _type_: _description_
    """
    y_predict_train = classif_model.predict(X_train)
    y_predict_test = classif_model.predict(X_test)
    y_proba_train = classif_model.predict_proba(X_train)[:,1]
    y_proba_test = classif_model.predict_proba(X_test)[:,1]
    metrics_train = get_metrics(y_true=y_train, y_pred=y_predict_train, y_pred_proba= y_proba_train, metrics_to_compute=metrics_to_compute)
    metrics_test = get_metrics(y_true=y_test,y_pred= y_predict_test, y_pred_proba = y_proba_test, metrics_to_compute=metrics_to_compute)
    res = {'train' : metrics_train, 'test' : metrics_test}
    return res

def classification_model(model, output_dim_model, 
                         sliding_window, stride,
                         datasetECG_train, datasetECG_val, datasetECG_test,
                         param_grid_svm, 
                         path_meta_data_train, path_meta_data_val, path_meta_data_test,
                         T = 1000,
                         metrics_to_compute = ['accuracy', 'f1_score','recall','precision'],
                         list_columns_classif = ['NORM'],
                         path_save_models= 'models/classif/',
                         apply_ICA = False):
    """Perform GridSearchCV using SVM and pretrained TCL model for all of the columns target in theprovided list
    Will save everything in repertories : 
    path_save_models+target_i/ with best_model.joblib , best_params.pkl, res_best_model.pkl, res_gs_SVM.pkl 

    Args:
        model (tcl): pretrained TCL model from TCL.py
        output_dim_model (int): n_components in TCL
        sliding_window (int): sliding window for average
        stride (int): stride for average
        datasetECG_train (ECGDataset): patients train
        datasetECG_val (ECGDataset): patients val
        datasetECG_test (ECGDataset): patients test
        param_grid_svm (dict): dict with keys as SVM parameters name in pipleine, values to test as keys
        path_meta_data_train (str): path for finding the meta data (target var) for patients in train
        path_meta_data_val (str): path for finding the meta data (target var) for patients in val
        path_meta_data_test (str): path for finding the meta data (target var) for patients in test
        T (int, optional): size of each component / initial signal . Defaults to 1000.
        metrics_to_compute (list, optional): Metrics to compute - chose between accuracy, recall, precision, f1_score, roc_auc. Defaults to ['accuracy', 'f1_score','recall','precision'].
        list_columns_classif (list, optional): list of target columns on which to perfrom classif. Defaults to ['NORM'].
        path_save_models (str, optional): directory path for saves. Defaults to 'models/classif/'.
        apply_ICA (bool) : False
    """
    df_train_meta = pd.read_csv(path_meta_data_train)
    df_val_meta = pd.read_csv(path_meta_data_val)
    df_test_meta = pd.read_csv(path_meta_data_test)
    id_patient_train, X_train_avg, X_features_train = get_features_dataset_parallel(model, output_dim_model, datasetECG_train, sliding_window, stride, T=T, apply_ICA=apply_ICA)
    with open(path_save_models+'train_info.pkl','wb') as f:
        d = {'id_patient':id_patient_train,
             'features_avg' : X_train_avg,
             'features_all' : X_features_train}
        pickle.dump(d,f)
    id_patient_val, X_val_avg, X_features_val = get_features_dataset_parallel(model, output_dim_model, datasetECG_val, sliding_window, stride, T=T, apply_ICA=apply_ICA)
    
    with open(path_save_models+'val_info.pkl','wb') as f:
        d = {'id_patient':id_patient_val,
             'features_avg' : X_val_avg,
             'features_all' : X_features_val}
        pickle.dump(d,f)  
    
    id_patient_test, X_test_avg, X_features_test = get_features_dataset_parallel(model, output_dim_model, datasetECG_test, sliding_window, stride, T=T, apply_ICA=apply_ICA)
    
    with open(path_save_models+'test_info.pkl','wb') as f:
        d = {'id_patient':id_patient_test,
             'features_avg' : X_test_avg,
             'features_all' : X_features_test}
        pickle.dump(d,f) 
    
    id_patient_train =  id_patient_train.numpy().astype(int)
    id_patient_val = id_patient_val.numpy().astype(int)
    id_patient_test = id_patient_test.numpy().astype(int)
    labels_train_dict = get_label_from_id_patient(id_patient_train, df_train_meta, list_columns_classif)
    labels_val_dict = get_label_from_id_patient(id_patient_val, df_val_meta, list_columns_classif)
    labels_test_dict = get_label_from_id_patient(id_patient_test, df_test_meta, list_columns_classif)
    X_train = X_train_avg.reshape(X_train_avg.shape[0], -1)
    X_val = X_val_avg.reshape(X_val_avg.shape[0], -1)
    X_test = X_test_avg.reshape(X_test_avg.shape[0], -1)
    X_big_train = np.concatenate([X_train,X_val], axis=0)
    all_res = {}
    for col in list_columns_classif :
        print('Training classifier for column {}'.format(col) ) 
        path_save_classif = path_save_models + col + '/'
        directory_path = os.path.dirname(path_save_classif)
        # Create directories if they don't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        y_train = labels_train_dict[col]
        y_train = y_train = y_train.ravel()
        y_val = labels_val_dict[col]
        y_val = y_val.ravel()
        y_test = labels_test_dict[col]
        y_test = y_test.ravel()
        y_big_train = np.concatenate([y_train,y_val], axis=0)
        results_gs, best_params, best_model = gridsearch_SVM_pipeline(X_train,y_train, X_val, y_val, param_grid_svm, scoring='accuracy', verbose=2)
        with open(path_save_classif+'results_gs_SVM.pkl','wb') as f :
            pickle.dump(results_gs,f)
        with open(path_save_classif+'best_params.pkl','wb') as f:
            pickle.dump(best_params,f)
        res = get_results(classif_model= best_model,
                          X_train = X_big_train,
                          y_train= y_big_train,
                          X_test= X_test, 
                          y_test= y_test,
                          metrics_to_compute= metrics_to_compute)
        with open(path_save_classif+'res_best_model.pkl','wb') as f:
            pickle.dump(res,f)
        model_filename = path_save_classif+'best_model.joblib'
        joblib.dump(best_model, model_filename)
        all_res[col] = res
    return(all_res)