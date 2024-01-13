import configparser
import os 
import json
import torch 

import pandas as pd
# Get the directory containing the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Set the working directory to the parent directory of the script (nonlinear_ICA)
os.chdir(os.path.abspath(os.path.join(script_dir, '..')))

from src.dataset import ECGSegmentedDataset, make_ECGsegmentedDataset_from_dataframe

if __name__ == "__main__":
    file_path = 'config/config_dataset.ini'
    config = configparser.ConfigParser(inline_comment_prefixes=(';',))
    config.read(file_path)
    params_all = config['dataset']
    n_dataset = params_all.getint('n_dataset')
    n_channels = params_all.getint('n_channels')
    path_save = params_all['path_save']
    df_train = pd.read_csv(params_all['path_data_train'])
    df_val = pd.read_csv(params_all['path_data_val'])
    df_test = pd.read_csv(params_all['path_data_test'])
    print('There are {} dataset to compute'.format(n_dataset))
    for i in range(1,n_dataset+1):
        path_save_dataset = path_save +'dataset_'+str(i)+'/'
        directory_path = os.path.dirname(path_save_dataset)
        # Create directories if they don't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        params_dataset_dict= dict(config.items('dataset_'+str(i)))
        params_dataset = config['dataset_'+str(i)]
        with open(path_save_dataset+'params_dataset.json','w') as f: 
            json.dump(params_dataset_dict,f)
        n_segment = params_dataset.getint('n_segment')
        n_point = params_dataset.getint('n_point_per_segment')
        dataset_train = make_ECGsegmentedDataset_from_dataframe(df=df_train,
                                                                n_channels= n_channels, 
                                                                n_segments = n_segment, 
                                                                n_point_per_segment = n_point)
        torch.save(dataset_train, path_save_dataset+'dataset_train.pth')
        dataset_val = make_ECGsegmentedDataset_from_dataframe(df=df_val,
                                                                n_channels= n_channels, 
                                                                n_segments = n_segment, 
                                                                n_point_per_segment = n_point)
        torch.save(dataset_val, path_save_dataset+'dataset_val.pth')
        dataset_test = make_ECGsegmentedDataset_from_dataframe(df=df_test,
                                                                n_channels= n_channels, 
                                                                n_segments = n_segment, 
                                                                n_point_per_segment = n_point)
        torch.save(dataset_test, path_save_dataset+'dataset_test.pth')
        print('dataset '+str(i)+' done')
