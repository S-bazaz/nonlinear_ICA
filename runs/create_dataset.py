# -*- coding: utf-8 -*-
##############
#  Packages  #
##############
import configparser
import sys
import os 
import json
import torch 
import pandas as pd
from pathlib import Path
##################
#      Imports   #
##################
# # Set the working directory to the parent directory of the script (nonlinear_ICA)
root_path = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, str(root_path))
config_path=root_path.joinpath("config", "config_dataset.ini")
from src.dataset import ECGSegmentedDataset, make_ECGsegmentedDataset_from_dataframe
###############
#    Script   #
###############

readcsv = lambda path: pd.read_csv(root_path.joinpath(path))

if __name__ == "__main__":

    config = configparser.ConfigParser(inline_comment_prefixes=(';',))
    config.read(config_path)

    #General
    params_all = config['dataset']
    n_dataset = params_all.getint('n_dataset')
    n_channels = params_all.getint('n_channels')
    path_save = root_path.joinpath(params_all['path_save'])
    
    #Loading
    df_train = readcsv(params_all['path_data_train'])
    df_val = readcsv(params_all['path_data_val'])
    df_test = readcsv(params_all['path_data_test'])

    print('There are {} dataset to compute'.format(n_dataset))
    for i in range(2,n_dataset+1): #1
        path_save_dataset = path_save.joinpath('dataset_'+str(i))
        directory_path = os.path.dirname(str(path_save_dataset)+"\\")
        # Create directories if they don't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        #creates paths
        train_path = str(path_save_dataset.joinpath('dataset_train.pth'))
        valid_path = str(path_save_dataset.joinpath('dataset_val.pth'))
        test_path =  str(path_save_dataset.joinpath('dataset_test.pth'))


        params_dataset_dict= dict(config.items('dataset_'+str(i)))
        params_dataset = config['dataset_'+str(i)]
        print("save json", str(path_save_dataset)+'\\params_dataset.json')
        with open(str(path_save_dataset)+'\\params_dataset.json','w') as f: 
            json.dump(params_dataset_dict,f)
    
        n_segment = params_dataset.getint('n_segment')
        n_point = params_dataset.getint('n_point_per_segment')
        dataset_train = make_ECGsegmentedDataset_from_dataframe(df=df_train,
                                                                n_channels= n_channels, 
                                                                n_segments = n_segment, 
                                                                n_point_per_segment = n_point)
        print(f"save train {train_path}")
        torch.save(dataset_train, train_path)
        
        dataset_val = make_ECGsegmentedDataset_from_dataframe(df=df_val,
                                                                n_channels= n_channels, 
                                                                n_segments = n_segment, 
                                                                n_point_per_segment = n_point)
        print(f"save validation {valid_path}")
        torch.save(dataset_val, valid_path)
        
        dataset_test = make_ECGsegmentedDataset_from_dataframe(df=df_test,
                                                                n_channels= n_channels, 
                                                                n_segments = n_segment, 
                                                                n_point_per_segment = n_point)
        print(f"save test {test_path}")
        torch.save(dataset_test, test_path)
        print('dataset '+str(i)+' done')
