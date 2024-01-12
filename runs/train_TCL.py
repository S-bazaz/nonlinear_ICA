
import torch 
import json
import configparser
import os 
import pickle

# Get the directory containing the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Set the working directory to the parent directory of the script (nonlinear_ICA)
os.chdir(os.path.abspath(os.path.join(script_dir, '..')))

import torch.optim as optim

from src.TCL import tcl, MLP, train
from src.dataset import ECGSegmentedDataset

from torch.utils.data import Dataset, DataLoader

from ast import literal_eval 

if __name__ == "__main__":
    file_path = 'config/config_TCL.ini'
    config = configparser.ConfigParser(inline_comment_prefixes=(';',))
    config.read(file_path)
    
    
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
    save_models = params_general.get('save_models')

    # Read Dataset parameters 
    params_dataset = config['dataset_info']
    dataset_train = torch.load(params_dataset["dataset_train"])
    #dataset_val = torch.load(params_dataset["dataset_val"])
    #dataset_test = torch.load(params_dataset["dataset_test"])
    n_batch_fake = params_dataset.getint('n_batch_stop')
    if n_batch_fake < 0 : 
        n_batch_fake = None
    n_segment = params_dataset.getint('n_segment')
    n_point = params_dataset.getint('n_point')
    n_patient = params_dataset.getint('n_patient')
    train_loader = DataLoader(dataset_train, batch_size=batch_train, shuffle=True)
    print('Loader train done')
    
    # Read TCL configurations
    tcl_configs = {}
    for i in range(1,n_models+1): 
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
        save_params_i = save_models + str(i)+ '/params.json'
        
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
        with open(save_params_i, 'w') as json_file:
            json.dump(save_params_i, json_file, indent=2)

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


   

