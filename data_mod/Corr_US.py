import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import random

import numpy
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm import tqdm

import args as ar
from data_mod.dataset import (ECGDataset_all, ECGDataset_few_shot,
                              ECGDataset_pair)
from utils import *

def data_gen_corr(y_data):
    def compute_corr_for_row(i):
        return sum([calc_correlation(y_data[i,: ], y_data[j, :]) for j in range( y_data.shape[0])])

    def calc_correlation(col1, col2):
        return np.corrcoef(col1, col2)[0, 1]
    # Use joblib to compute upper triangle in parallel
    n_jobs = -1  # Use all available cores
    corr_index = Parallel(n_jobs=n_jobs)(
    delayed( compute_corr_for_row)(i) for i in range( y_data.shape[0]))
    return corr_index

if __name__=="__main__":
    arg = ar.parse_args()
    datadir=arg.data_dir
    basepath="mitbih_all"
    base_path="./data/"+basepath
    random.seed(arg.seed)
    for i in range(10):
        train_data_path=os.path.join(base_path,"train","data",basepath+"_fold"+str(i)+".npy")
        train_label_path=os.path.join(base_path,"train","label",basepath+"_fold"+str(i)+".npy")
        train_dataset=ECGDataset_all(train_data_path,train_label_path)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True,prefetch_factor=3)
        output_list=[]
        labels_list=[]
        for _, (data, labels) in enumerate(tqdm(train_loader)):
            output_list.append(data)
            labels_list.append(labels)
        y_data = np.vstack(output_list)
        y_label = np.vstack(labels_list)
        y_data=np.squeeze(y_data, axis=1)
        y_label=np.squeeze(y_label,axis=None)
        print(y_label.shape)
        print(y_data.shape)
        label_class_data=np.argmax(y_label,1)
        correlation_index=np.zeros(y_data.shape[0])
        for c in range(5):
            label_indices=np.where(label_class_data==c)[0]
            correlation_index[label_indices[0:2]]=data_gen_corr(y_data[label_indices[0:2]])
        save_path=os.path.join(base_path,"train","corr_index",basepath+"_fold"+str(i)+".npy")
        np.save(save_path,correlation_index)
    
    




