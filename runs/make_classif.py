# -*- coding: utf-8 -*-
##############
#  Packages  #
##############
import torch 
import json
import configparser
import sys
import os 
import pickle
import ast
from tqdm import tqdm
from pathlib import Path
################
#    Imports   #
################
# # Set the working directory to the parent directory of the script (nonlinear_ICA)
root_path = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, str(root_path))
config_path=root_path.joinpath("config", "config_classif.ini")

from src.TCL import tcl, MLP, Maxout
from src.feature_extraction import get_features_dataset, get_label_from_id_patient
from src.classification import classification_model

###############
#    Script   #
###############
join_root = lambda meta_path: str(root_path.joinpath(meta_path))
load_data = lambda dataset_path, type: torch.load(str(root_path.joinpath(dataset_path, f'dataset_{type}.pth')))

if __name__ == "__main__":
    config = configparser.ConfigParser(inline_comment_prefixes=(';',))
    config.read(config_path)
    ### TCL info 
    TCL_info = ast.literal_eval(config['TCL']['TCL_info'])
    n_sw = config['TCL'].getint('n_possibilities_windows_stride')
    sliding_windows_stride = ast.literal_eval(config['TCL']['params_sliding_windows_stride'])
    apply_ICA = ast.literal_eval(config['TCL']['apply_ICA'])
    ## Dataset info 
    datasets = config['datasets']
    datasets_info = ast.literal_eval(datasets['datasets_info'])
    T = datasets.getint('T')
    ### Meta Data info 
    meta_data = config['meta_data']
    path_meta_train = join_root(meta_data['train'])
    path_meta_val = join_root(meta_data['val'])
    path_meta_test = join_root(meta_data['test'])
    ### Classif info
    classif = config['classif']
    param_grid_svm = ast.literal_eval(classif['param_grid'])
    columns_to_classif = ast.literal_eval(classif['columns_classif'])
    path_save_classif = join_root(classif['path_save_models'])
    metrics_to_compute = ast.literal_eval(classif['metrics_to_compute'])
    for i in tqdm(range(n_sw)):
        name_classif = 'classif_'+str(i+1)
        window = sliding_windows_stride[i][0]
        stride = sliding_windows_stride[i][1]
        params_classif_TCL = {'window':window, 
                              'stride': stride}
        path_save_classif_i = path_save_classif + '/classif_'+str(i+1)+'/'

        directory_path = os.path.dirname(path_save_classif_i)
        # Create directories if they don't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(path_save_classif_i+'params_window_stride_agreg.json','w') as json_file:
            json.dump(params_classif_TCL,json_file, indent=2)
        for dataset_name, dataset_path in datasets_info.items():               

            datasetECG_train = load_data(dataset_path,"train")
            datasetECG_val = load_data(dataset_path,"val")
            datasetECG_test = load_data(dataset_path,"test")
            path_save_classif_i_dataset_j = path_save_classif_i + dataset_name +'/'

            for TCL_name, TCL_k_pretrained_info in TCL_info.items():
                path_save_classif_i_dataset_j_TCL_k = path_save_classif_i_dataset_j+ TCL_name +'/'
                directory_path = os.path.dirname(path_save_classif_i_dataset_j_TCL_k)
                # Create directories if they don't exist
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                TCL_k = torch.load(join_root(TCL_k_pretrained_info['path']))
                output_dim_model_k = TCL_k_pretrained_info['output_dim']
                res = classification_model(model = TCL_k, 
                                           output_dim_model = output_dim_model_k, 
                                            sliding_window = window,
                                            stride = stride,
                                            datasetECG_train = datasetECG_train,
                                            datasetECG_val = datasetECG_val,
                                            datasetECG_test = datasetECG_test,
                                            param_grid_svm = param_grid_svm, 
                                            path_meta_data_train = path_meta_train,
                                            path_meta_data_val = path_meta_val, 
                                            path_meta_data_test = path_meta_test,
                                            T = T,
                                            metrics_to_compute = metrics_to_compute,
                                            list_columns_classif = columns_to_classif,
                                            path_save_models = path_save_classif_i_dataset_j_TCL_k,
                                            apply_ICA= apply_ICA)
                print("Save_results, ",path_save_classif_i_dataset_j_TCL_k+'results_all_targets.pkl')
                with open(path_save_classif_i_dataset_j_TCL_k+'results_all_targets.pkl','wb') as f: 
                    pickle.dump(res,f)
                
