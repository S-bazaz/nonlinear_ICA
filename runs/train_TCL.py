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
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval 

################
#    Imports   #
################
# # Set the working directory to the parent directory of the script (nonlinear_ICA)
root_path = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, str(root_path))
config_path=root_path.joinpath("config", "config_TCL.ini")
data_set_path = root_path.joinpath("config", "config_dataset.ini")

from src.TCL import tcl, MLP, train
from src.dataset import ECGSegmentedDataset

###############
#    Script   #
###############

if __name__ == "__main__":
    config = configparser.ConfigParser(inline_comment_prefixes=(';',))
    config.read(config_path)
    data_config  = configparser.ConfigParser(inline_comment_prefixes=(';',))
    data_config.read(data_set_path)
    
    # Read general parameters
    params_general = config['general']
    n_models = params_general.getint('n_models')
    random_seed = params_general.getint('random_seed')
    n_epochs = params_general.getint('n_epochs')
    batch_train = params_general.getint('batch_train')
    lr = params_general.getfloat('learning_rate')
    weight_decay = params_general.getfloat('weight_decay')
    step_size = params_general.getint('step_size')
    gamma = params_general.getfloat('gamma')
    save_models = str(root_path.joinpath(params_general.get('save_models')))
    

    # Read Dataset parameters 
    params_dataset = config['dataset_info']
    dataset_idx = params_dataset.getint('data_idx')

    dataset_id = f"dataset_{dataset_idx}"
    params_data = data_config[dataset_id]
    n_segment = params_data.getint('n_segment')
    n_point = params_data.getint('n_point_per_segment')

    #train set loading
    load_params = data_config["dataset"]
    train_path = str(root_path.joinpath(load_params.get("path_save"), dataset_id ,"dataset_train.pth"))
    print(f"Train TCL from {train_path}")
    dataset_train = torch.load(train_path)


    n_batch_fake = params_dataset.getint('n_batch_stop')
    if n_batch_fake < 0 : 
        n_batch_fake = None
        
    # n_segment = params_dataset.getint('n_segment')
    # n_point = params_dataset.getint('n_point')
    n_patient = params_dataset.getint('n_patient')
    train_loader = DataLoader(dataset_train, batch_size=batch_train, shuffle=True)
    print('Loader train done')
    
    # Read TCL configurations
    tcl_configs = {}
    for i in range(5,n_models+1): 
        name_i = 'TCL_'+str(i)
        print(name_i)
        tcl_i = config[name_i]
        params_tcl_i = {} 
        num_layers = tcl_i.getint('num_layers')
        hidden_layers = literal_eval(tcl_i.get('hidden_layers'))
        activation = literal_eval(tcl_i.get('activation'))
        input_dim = tcl_i.getint('input_dim')
        pool_size = tcl_i.getint('pool_size')
        slope = tcl_i.getfloat('slope')
        save_model_i = save_models + str(i)+ '/model'
        save_params_i = save_models + str(i)+ '/params.pkl'
        print(f"Save model {save_models}")
        
        directory_path = os.path.dirname(save_params_i)
        # Create directories if they don't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        params_tcl_i = {'num_layers':num_layers,
                        'hidden_layers' : hidden_layers, 
                        'activation' : activation, 
                        'input_dim' : input_dim,
                        'pool_size' : pool_size, 
                        'slope' : slope
                        }
        with open(save_params_i, 'wb') as file:
            pickle.dump(params_tcl_i, file)

        tcl_configs[name_i] = tcl_i
        model = tcl(input_dim=input_dim,
                    hidden_dim=hidden_layers,
                    num_class=n_segment, 
                    num_patient=n_patient+1,
                    num_layers=num_layers)
        optimizer = optim.SGD(model.parameters(), 
                              lr=lr, 
                              weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=step_size, 
                                              gamma=gamma)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model, train_loss = train(model = model,
                                  loader_train = train_loader,  
                                  optimizer= optimizer, 
                                  scheduler = scheduler,
                                  n_epochs = n_epochs,
                                  device = device, 
                                  num_batch_to_process= n_batch_fake,
                                  file_path= save_model_i)
        with open(save_model_i+'_loss.pkl', 'wb') as f:
            pickle.dump(train_loss, f)
        
        torch.save(model,save_model_i+'.pth')


   

