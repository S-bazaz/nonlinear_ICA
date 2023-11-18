# -*- coding: utf-8 -*-
##############
#  Packages  #
##############
import os
import sys
import wfdb
import ast
import numpy as np
import pandas as pd

from pathlib import Path

from pandas.api.types import is_hashable
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from typing import Dict, Optional, Union, BinaryIO, Tuple, List

##################
#      Imports   #
##################
root_path = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, str(root_path))

################
#   loading    #
################

def load_meta(root_path):
    """
    Load metadata related to electrocardiogram (ECG) data.

    Args:
        root_path (str or Path): Root path containing the 'raw_data' directory.

    Returns:
        dict: Dictionary containing loaded dataframes.
            - 'ecg_meta': Metadata related to ECG data.
            - 'scp': SCP statements dataframe.
    """
    
    meta_path = root_path.joinpath("raw_data", "meta_data")
    ecg_path = str(meta_path.joinpath('ptbxl_database.csv'))
    scp_path = str(meta_path.joinpath('scp_statements.csv'))
    
    df_ecg_meta = pd.read_csv(ecg_path, index_col='ecg_id')
    df_ecg_meta.scp_codes = df_ecg_meta.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    return  {"ecg_meta": df_ecg_meta,
              "scp": pd.read_csv(scp_path, index_col=0)
              }
    
def load_ecg(dct_data, root_path, sampling_rate=100, ecg_ids=None):
    """
    Load electrocardiogram (ECG) data based on metadata.

    Args:
        dct_data (dict): Dictionary containing data.
        root_path (str or Path): Root path containing the 'raw_data' directory.
        sampling_rate (int): Sampling rate of the ECG data.
        ecg_ids (list or None): List of ECG IDs to load. If None, load all.

    Returns:
        None: Updates the 'ecg' key in dct_data with loaded ECG data.
    """
    df_ecg_meta = dct_data["ecg_meta"]
    ecg_path = root_path.joinpath("raw_data","ecg_data")
    
    if sampling_rate == 100:
        col = "filename_lr"
    else:
        col = "filename_hr"
    if ecg_ids:
        filenames = df_ecg_meta.loc[ecg_ids, col]
    else:
        filenames = df_ecg_meta[col]
    
    print(filenames)
    load_signal = lambda path : wfdb.rdsamp(
        str(ecg_path.joinpath(path))
        )[0] 

    
    dct_data["ecg"] = filenames.apply(load_signal).to_numpy()




#######################
#   get superclass    #
#######################

def aggregate_diagnostic(agg_df, y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def add_superclasse(dct_data):
    agg_df = dct_data["scp"].copy()
    agg_df = agg_df[agg_df.diagnostic == 1]
    f_agg = lambda y_dic: aggregate_diagnostic(agg_df, y_dic)
    dct_data["ecg_meta"]['diagnostic_superclass'] = dct_data["ecg_meta"].scp_codes.apply(f_agg)
    


############################
#   raw data description   #
############################

def df_nan_percent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le pourcentage de valeurs manquantes dans chaque colonne d'un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.

    Returns:
        pd.DataFrame: DataFrame contenant le pourcentage de valeurs manquantes pour chaque colonne.
    """
    part_nan = df.isnull().mean()
    perc_nan = pd.DataFrame(np.round(100 * part_nan.to_numpy(), 1))
    perc_nan.index = part_nan.index
    return perc_nan

def describ_raw_df(df):
    print("\n---------Shape------------------\n", df.shape)
    print("\n---------Types------------------\n", df.dtypes)
    print("\n--------- 1 row-----------------\n", df.iloc[0, :])
    print("\n---------describe---------------\n", df.describe())
    print("\n---------nan percentage---------\n", df_nan_percent(df))
    for col in df:
        unique_val = df[col].explode().unique()
        if len(unique_val) < 20:
            print(f"\n---------{col} values----------")
            print(unique_val)
            

def plot_all_st(X, clustering=None, title="<b>Signals</b>"):
    """
    Plot multiple signals in a single interactive Plotly figure.

    Args:
        X (list of arrays): A list of signal arrays to be plotted.
        clustering (list or None, optional): A list of cluster assignments for each signal. 
            If provided, signals will be color-coded by cluster. Default is None.
        title (str, optional): The title of the plot. Default is "<b>Signals</b>".

    Returns:
        None: Displays an interactive Plotly figure with the plotted signals.
    """
    
    fig = go.Figure(
        layout=go.Layout(
            height=600, 
            width=800, 
            template = "plotly_dark", 
            title = title
    ))
        
    if clustering:
        pal = ["palegreen", "darkred"]
    else:
        pal = sns.color_palette("Spectral", len(X)).as_hex()

    for i in range(len(X)):
        if clustering:
            color = pal[clustering[i]]
        else:
            color = pal[i]
            
        fig.add_trace(go.Scatter(y=X[i], 
                                 mode="lines", 
                                 line=dict(
                                     width=2,
                                     color=color,
                                 ),
                                 opacity = 0.6
                                ))
    fig.show()