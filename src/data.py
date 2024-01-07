# -*- coding: utf-8 -*-
##############
#  Packages  #
##############
import os
import sys
import pywt
import pickle
import numpy as np
import pandas as pd
from numba import jit
from pathlib import Path
from sklearn.decomposition import PCA
from pandas.api.types import is_hashable

from typing import Dict, Optional, Union, BinaryIO, Tuple, List

##################
#      Imports   #
##################
root_path = Path(os.path.abspath(__file__)).parents[1]
sys.path.insert(0, str(root_path))
data_path = root_path.joinpath("raw_data")

##################
#   denoising    #
##################

def denoise_1sign(sign):
    #decomposition en ondelette
    coeffs = pywt.wavedec(sign, "sym4", mode="per")
    #filtration qui s'apparente à une projection orthogonale 
    coeffs[1:] = (pywt.threshold(coeff, value=0.2, mode="soft") for coeff in coeffs[1:])
    #reconstruction du signal filtré
    return pywt.waverec(coeffs, "sym4", mode="per" )

########################
#   ACP et pipeline    #
########################

def pca_tens(tens, ncomp=5):
    # initialisation du tenseur réduit
    shape = list(tens.shape)
    shape[2] = ncomp
    pca_tens = np.zeros(shape)
    # ACP pour chaque ecg_id
    pca = PCA(n_components=ncomp)
    ratios = np.zeros((shape[0], ncomp))
    for i,mat in enumerate(tens):
        pca_tens[i,:,:] = pca.fit_transform(mat)
        ratios[i,:] = np.array(pca.explained_variance_ratio_)
    print("PCA mean ratios",np.mean(ratios, axis = 0))
    return pca_tens

def denoise_and_pca(df, ncomp=5):
    #reshape à 3D avec la première dimension associée à l'identifiant de l'ecg
    tens = df.iloc[:,1:].to_numpy().reshape(-1,1000,12)
    #on applique le denoising sur chaque signaux dont l'abcisse est associé à la dimension 1
    tens = np.apply_along_axis(denoise_1sign, arr = tens, axis = 1 )
    # on termine avec une ACP pour passer de 12 canaux à "ncomp" signaux 
    # de manière à éliminer les redondances entre électrodes
    return  pca_tens(tens,ncomp=ncomp)

########################
#   Final functions    #
########################

def csv_to_processed_pickle(data_path, name="train", ncomp=5): 
    #name = train, valid, ou test

    #loading
    df = pd.read_csv(data_path.joinpath(f"{name}_signal.csv"))
    #processing
    tens = denoise_and_pca(df, ncomp=4)
    #ids to connect tens with meta data
    ecg_ids = np.unique(df["ecg_id"])
    #saving
    with open(str(data_path.joinpath(f"{name}_tens.pkl")), "wb") as f:
        pickle.dump(tens, f)
    with open(str(data_path.joinpath(f"{name}_ids.pkl")), "wb") as f:
        pickle.dump(ecg_ids, f)


def loader(data_path, name="train"):
    with open(str(data_path.joinpath(f"{name}_tens.pkl")), "rb") as f:
        tens = pickle.load(f)
    with open(str(data_path.joinpath(f"{name}_ids.pkl")), "rb") as f:
        ecg_ids = pickle.load(f)
    df_meta = pd.read_csv(data_path.joinpath(f"{name}_meta.csv"))
    return tens, ecg_ids, df_meta


###############
#   Script    #
###############

csv_to_processed_pickle(data_path, name="train", ncomp=5)


#############
#   test    #
#############
    

#plotting_________________
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
pio.renderers.default = "browser"  
def my_pal(n):
    return sns.color_palette("Spectral", n).as_hex()

def plot_signal(vec, title = "signal"):
    fig = px.line(vec, template = "plotly_dark", title = title)
    fig.show()

def add_fig(fig, signal, color, name):
    fig.add_trace(go.Scatter(y=signal, 
                 mode="lines", 
                 line=dict(
                     width=2,
                     color=color,
                 ),
                 opacity = 0.6,
                 name=name
                )
             )

    
#loading_________________
tens_train, ids_train, meta_train = loader(data_path, name="train")

#les identifiants ici sont dans même ordre
print(tens_train.shape)
print(ids_train.shape)

pal = my_pal(8)    
for i, mat in enumerate(tens_train[:10]):
    fig = go.Figure(
        layout=go.Layout(
            height=600, 
            width=800, 
            template = "plotly_dark", 
            title = f"ACP ecg_id = {ids_train[i]}"
    ))

    for j, sign in enumerate(mat.T):
        color = pal[j]
            
        fig.add_trace(go.Scatter(y=sign, 
                            mode="lines", 
                            line=dict(
                                width=2,
                                color=color,
                            ),
                            opacity = 0.6
                        ))  
    fig.show()