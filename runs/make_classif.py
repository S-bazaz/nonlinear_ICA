
import torch 
import json
import configparser
import os 
import pickle
import ast

# Get the directory containing the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Set the working directory to the parent directory of the script (nonlinear_ICA)
os.chdir(os.path.abspath(os.path.join(script_dir, '..')))

from src.TCL import tcl, MLP, Maxout
from src.feature_extraction import get_features_dataset, get_label_from_id_patient
from src.classification import classification_model

if __name__ == "__main__":
    file_path = 'config/config_classif.ini'
    config = configparser.ConfigParser(inline_comment_prefixes=(';',))
    config.read(file_path)
    ### TCL info 
    TCL_info = ast.literal_eval(config['TCL']['TCL_info'])
    n_sw = config['TCL'].getint('n_possibilities_windows_stride')
    sliding_windows_stride = ast.literal_eval(config['TCL']['params_sliding_windows_stride'])
    ## Dataset info 
    datasets = config['datasets']
    datasets_info = ast.literal_eval(datasets['datasets_info'])
    T = datasets.getint('T')
    ### Meta Data info 
    meta_data = config['meta_data']
    path_meta_train = meta_data['train']
    path_meta_val = meta_data['val']
    path_meta_test = meta_data['test']
    ### Classif info
    classif = config['classif']
    param_grid_svm = ast.literal_eval(classif['param_grid'])
    columns_to_classif = ast.literal_eval(classif['columns_classif'])
    path_save_classif = classif['path_save_models']
    metrics_to_compute = ast.literal_eval(classif['metrics_to_compute'])
    for i in range(n_sw):
        name_classif = 'classif_'+str(i+1)
        window = sliding_windows_stride[i][0]
        stride = sliding_windows_stride[i][1]
        params_classif_TCL = {'window':window, 
                              'stride': stride}
        path_save_classif_i = path_save_classif + 'classif_'+str(i+1)+'/'
        directory_path = os.path.dirname(path_save_classif_i)
        # Create directories if they don't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(path_save_classif_i+'params_window_stride_agreg.json','w') as json_file:
            json.dump(params_classif_TCL,json_file, indent=2)
        for dataset_name, dataset_path in datasets_info.items():               
            datasetECG_train = torch.load(dataset_path+'dataset_train.pth')
            datasetECG_val = torch.load(dataset_path+'dataset_val.pth')
            datasetECG_test = torch.load(dataset_path+'dataset_test.pth')
            path_save_classif_i_dataset_j = path_save_classif_i + dataset_name +'/'

            for TCL_name, TCL_k_pretrained_info in TCL_info.items():
                path_save_classif_i_dataset_j_TCL_k = path_save_classif_i_dataset_j+ TCL_name +'/'
                directory_path = os.path.dirname(path_save_classif_i_dataset_j_TCL_k)
                # Create directories if they don't exist
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                TCL_k = torch.load(TCL_k_pretrained_info['path'])
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
                                            path_save_models = path_save_classif_i_dataset_j_TCL_k )
                with open(path_save_classif_i_dataset_j_TCL_k+'results_all_targets.pkl','wb') as f: 
                    pickle.dump(res,f)
                
