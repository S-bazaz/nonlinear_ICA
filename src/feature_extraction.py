import json
import numpy as np
import torch
from sklearn.decomposition import FastICA
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
#UNCOMMENT FOR TRAIN
# if torch.cuda.is_available():
#     device = 'cuda:0'
# else:
#     device = 'cpu'

#UNCOMMENT FOR TEST
device = 'cpu'

def get_number_windows(shape, w, s):
    """Given a certain length, compute size of output (number of windows)
    when input of size length is given and
    then averaged on sliding window w with stride s

    Args:
        shape (int): length of input
        w (int): size of window
        s (int): size of stride
    """
    num_windows = (shape - w) // s + 1
    return(num_windows)

def compute_average_signal(original_tensor, w, s):
    """Given a tensor of shape [n,L] will average on L using sliding window w and stride s

    Args:
        original_tensor (tensor): of shape (n,L)
        w (int): sliding window
        s (int): stride
    """
    # Use unfold to create overlapping windows
    unfolded_tensor = original_tensor.unfold(1, w, s)

    # Average across the window dimension
    averaged_tensor = unfolded_tensor.mean(dim=2)
    
    return(averaged_tensor)

def get_features_dataset(model, output_dim, dataset, sliding_window, stride, T=1000, 
                         apply_ICA = False):
    """Given a pretrained TCL model with output dim (n_components) and a certain dataset
    will compute sliding window averaging for every channel of ouptut_dim 
    it is done by patient so the output for a dataset with n patients is 
    (tensor_id_patient of size n, feature_patient of size [n, n_components, n_windows])

    Args:
        model (TCL model): TCL model trained from tcl in TCL.py
        output_dim (int): n_components after TCL
        dataset (ECGDataset): ECG Dataset
        sliding_window (int): size of sliding_windows
        stride (int): stride for averaging
        T (int, optional): Length of every channel. Defaults to 1000.
    """
    batch_size = T 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_patient_dataset = len(loader)
    i=0
    num_windows = get_number_windows(T,sliding_window,stride)
    label_patients = np.zeros(n_patient_dataset)
    features_patients_avg = np.zeros((n_patient_dataset,output_dim, num_windows ))
    features_patients_all = np.zeros((n_patient_dataset,output_dim, T))
    for batch_x, batch_y, batch_z in loader : 
        output_patient_logit, features_patient = model(batch_x.float(), patient_id = batch_z)
        patient_id = batch_z.unique()
        assert patient_id.shape[0]==1, 'Multiple patient in batch - resize batch size'
        label_patients[i] = patient_id[0]
        if apply_ICA:
            transformer = FastICA(n_components=output_dim,random_state=0, whiten='unit-variance')
            X = features_patient.detach().numpy()
            X_transformed = transformer.fit_transform(X)
            features_patient = torch.tensor(X_transformed)
        features_patient_avg = compute_average_signal(features_patient.T, w=sliding_window, s=stride)
        features_patients_all[i,:,:]= features_patient.T
        features_patients_avg[i,:,:] = features_patient_avg.detach().numpy()
        i+=1 
    index_sort = np.argsort(label_patients)
    label_patients = label_patients[index_sort]
    features_patients_avg = features_patients_avg[index_sort,:,:]
    features_patients_all = features_patients_all[index_sort,:,:]
    return(label_patients, features_patients_avg, features_patients_all)
    
def get_label_from_id_patient(id_patient, df_meta, list_columns):
    """Given a list of patient and a df with patients and columns
    will give labels for each patient and each column

    Args:
        id_patient (array): with id of patients
        df_meta (DataFrame): containing all patient from id_patient and columns in list_columns
        list_columns (list): list of target variables
    """
    df_ = df_meta.sort_values(by=['ecg_id'], ascending = True)
    id_in_meta = np.array(df_['ecg_id'].values, dtype=int)
    nb_label_per_patient = df_['ecg_id'].value_counts().values
    unique_values = len(np.unique(nb_label_per_patient))
    assert unique_values == 1 , 'Error: some patients have multi labels'
    assert np.isin(id_patient, id_in_meta).all(), 'Error : some patients are not included in dataframe provided'
    d = {}
    df_subset = df_.loc[df_['ecg_id'].isin(id_patient),:]
    for col in list_columns:
        d[col] = df_subset[col].values
    return(d)


# these 2 functions are just the parallelized version of get_features_dataset
def process_batch(batch, model, output_dim, sliding_window, stride, apply_ICA=False):
    batch_x, batch_y, batch_z = batch
    output_patient_logit, features_patient = model(batch_x.float(), patient_id=batch_z)
    patient_id = batch_z.unique()
    assert patient_id.shape[0] == 1, 'Multiple patients in batch - resize batch size'
    if apply_ICA:
            transformer = FastICA(n_components=output_dim,random_state=0, whiten='unit-variance')
            X = features_patient.detach().numpy()
            X_transformed = transformer.fit_transform(X)
            features_patient = torch.tensor(X_transformed)
    features_patient_avg = compute_average_signal(features_patient.T, w=sliding_window, s=stride)
    return patient_id[0], features_patient_avg.detach().numpy(), features_patient

def get_features_dataset_parallel(model, output_dim, dataset, sliding_window, stride, T=1000, apply_ICA=False):
    batch_size = T
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_patient_dataset = len(loader)

    label_patients = np.zeros(n_patient_dataset)
    features_patients_avg = np.zeros((n_patient_dataset, output_dim, get_number_windows(T, sliding_window, stride)))
    features_patients_all = np.zeros((n_patient_dataset, output_dim, T))
    with ThreadPoolExecutor() as executor:
        batches = list(loader)
        results = list(executor.map(lambda batch: process_batch(batch, model, output_dim, sliding_window, stride, apply_ICA=apply_ICA), batches))

    for i, (patient_id, features_patient_avg, features_patient) in enumerate(sorted(results, key=lambda x: x[0])):
        label_patients[i] = patient_id
        features_patients_avg[i, :, :] = features_patient_avg
        features_patients_all[i,:,:] = features_patient.T
    # return label_patients, features_patients_avg, features_patients_all
    return (torch.tensor(label_patients,device=device), 
        torch.tensor(features_patients_avg,device=device),
        torch.tensor(features_patients_all,device=device),
    )

