## Nonlinear ICA
Our objective is to study feature extraction techniques for multidimensional time series data taken from PTB-XL-Dataset.
please dowload the data at *https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset-reformatted/data* and store the csv files into *data\raw_data\csv_data\*

### Wavelet filtration selection

The ondellette filtration study is present in the notebook *0_data_denoising.ipynb*  which motivates the choice of soft thresoholding with Symlet4 used in the pre-processing pipeline.

![mode_soft](https://github.com/S-bazaz/nonlinear_ICA/assets/108877488/7b861bac-92e6-40e4-80a1-09a220c366e5)

### Classification pipelines

In all the following steps, change the configuration using the .ini files in *config\*, then run the associated script in *runs\*. 
If an issue occure for instance concerning devices conflict, please manage the device definition in *src\* files.

#### dataset creation

Manage the segmentation of data in config_dataset.ini to create the datasets used to train and test the models, and run create_dataset.py

#### TCL train

Manage the TCL architectures in config_TCL.ini and run train_TCL.py

#### SVM Classification

Manage the ICA subsampling and SVM grid search in config_classif.ini and run make_classif.py
ICA is slow, so reduce the data by 50 if you don't have a week or a cluster.

#### Visualization of the results

Use *1_classification_results.ipynb* to see the metrics and ICAs.
TCL_ids and classif_ids have to be adapted to the classification you made.

![TCL2_ICA_sub1](https://github.com/S-bazaz/nonlinear_ICA/assets/108877488/1ef11a3d-0a70-49a9-8c82-2b12f2edf8d3)

Enjoy this non linear source separation method !
