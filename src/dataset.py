import torch
import numpy as np
import tqdm

from torch.utils.data import Dataset, DataLoader

import torch
import os

class ECGSegmentedDataset(Dataset):
    def __init__(self, seg_data, seg_label, seg_patient_id):
        self.seg_data = seg_data
        self.seg_label = seg_label
        self.seg_patient_id = seg_patient_id

    def __len__(self):
        return self.seg_data.shape[0]

    def __getitem__(self, idx):
        data_idx = torch.tensor(self.seg_data[idx, :])
        label_idx = torch.tensor(self.seg_label[idx])
        patient_idx = torch.tensor(self.seg_patient_id[idx])
        return data_idx, label_idx, patient_idx

def dict_dataset_per_patient_old(df,
                            n_channels, 
                            column_id = 'ecg_id',
                            prefix_channel='channel-'):
    channels_names = [prefix_channel+str(i) for i in range(n_channels)]
    ids =df[column_id].unique()
    dict_info= {}
    for id in tqdm.tqdm(ids, desc='Select sub df'):
        dict_info[id]= df.loc[df[column_id]==id, channels_names].values
    return(dict_info)

def dict_dataset_per_patient(df, n_channels, column_id='ecg_id', prefix_channel='channel-'):
    channels_names = [prefix_channel + str(i) for i in range(n_channels)]
    
    # Group by 'ecg_id' and aggregate the data into a dictionary
    grouped_data = df.groupby(column_id)[channels_names].apply(lambda x: x.values).to_dict()

    return grouped_data

def segment_dataset(dict_data,
                    n_segments, 
                    points_per_segment):
    #assert that channels_name are in df.columns
    data_ex = list(dict_data.values())[0]
    assert n_segments*points_per_segment == data_ex.shape[0], "Number of segments times number point per segment is not compatible ith number of points per channel"
    all_points = []
    all_labels = []
    all_patient_id = []
    for id, data in  dict_data.items():
        id_point_beginning = 0
        id_point_end = points_per_segment
        for seg in range(n_segments):
            points_seg = data[id_point_beginning:id_point_end,:]
            label = np.zeros(points_per_segment)+seg
            patient_id = np.zeros(points_per_segment)+id
            all_points.append(points_seg)
            all_labels.append(label)
            all_patient_id.append(patient_id)
            id_point_beginning = id_point_end
            id_point_end = id_point_end + points_per_segment
    all_points = np.concatenate(all_points, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_patient_id = np.concatenate(all_patient_id, axis=0)
    return(all_points, all_labels, all_patient_id)

def segment_dataset_2(dict_data, n_segments, points_per_segment):
    data_ex = list(dict_data.values())[0]
    assert n_segments * points_per_segment == data_ex.shape[0], "Number of segments times number point per segment is not compatible with the number of points per channel"

    all_points = []
    all_labels = []
    all_patient_id = []

    for id, data in tqdm.tqdm(dict_data.items(),total=len(dict_data),desc="segment_data"):
        # Use numpy's reshape to segment the data more efficiently
        data_segments = data.reshape(n_segments * points_per_segment, data.shape[1])
    
        # Create labels and patient IDs
        labels = np.repeat(np.arange(n_segments),points_per_segment)
        patient_ids = np.full_like(labels, fill_value=id)

        all_points.append(data_segments)
        all_labels.append(labels)
        all_patient_id.append(patient_ids)

    all_points = np.concatenate(all_points, axis=0)  # Concatenate along the points_per_segment dimension
    all_labels = np.concatenate(all_labels)
    all_patient_id = np.concatenate(all_patient_id)

    return all_points, all_labels, all_patient_id


    


def make_ECGsegmentedDataset_from_dataframe(df, n_channels, n_segments, n_point_per_segment):
    dict_df = dict_dataset_per_patient(df, n_channels)
    #seg_data, seg_label, seg_patient_id = segment_dataset(dict_df,n_segments=n_segments,points_per_segment=n_point_per_segment)
    seg_data, seg_label, seg_patient_id = segment_dataset_2(dict_df,n_segments=n_segments,points_per_segment=n_point_per_segment)
    dataset = ECGSegmentedDataset(seg_data=seg_data,
                                    seg_label=seg_label,
                                    seg_patient_id=seg_patient_id)
    return(dataset)
    


