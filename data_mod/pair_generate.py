import sys
import os 
from pathlib import Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from args import parse_args
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from data_mod.dataset import ECGDataset_all
from sklearn.model_selection import train_test_split
import random
from itertools import combinations

def folder_generate(name):
    data_diff=["1","5","10","30","50","90","150"]

    for diff in data_diff:
        path=Path("./data",name+"_"+diff+"_"+"pair")
        if not path.exists():
            path.mkdir()
            train_path=Path(path,"train")
            train_path.mkdir()
            train_path_sub=Path(train_path,"data")
            train_path_sub.mkdir()
            train_path_sub=Path(train_path,"label")
            train_path_sub.mkdir()
            test_path=Path(path,"test")
            test_path.mkdir()
            test_path_sub=Path(test_path,"data")
            test_path_sub.mkdir()
            test_path_sub=Path(test_path,"label")
            test_path_sub.mkdir()
            val_path=Path(path,"val")
            val_path.mkdir()
            val_path_sub=Path(val_path,"data")
            val_path_sub.mkdir()
            val_path_sub=Path(val_path,"label")
            val_path_sub.mkdir()
        else:
            print("Dir exit.")
        path=Path("./result",name+"_"+diff)
        if not path.exists():
            os.makedirs(path)
        else:
            print("Dir exit.")

def pair_datagen(arg,oripath,folds):
    data_diff=["1","5","10","30","50","90","150"]
    for diff in data_diff:
        ori_path=oripath+"_"+diff
        train_data_path=os.path.join("./data", ori_path,"train","data",ori_path+"_fold"+str(folds)+".npy")
        train_label_path=os.path.join("./data", ori_path,"train","label",ori_path+"_fold"+str(folds)+".npy")
        train_data_path_save_1=os.path.join("./data", ori_path+"_pair","train","data",ori_path+"_pair1_fold"+str(folds)+".npy")
        train_data_path_save_2=os.path.join("./data",ori_path+"_pair","train","data",ori_path+"_pair2_fold"+str(folds)+".npy")
        train_label_path_save=os.path.join("./data",ori_path+"_pair","train","label",ori_path+"_pair_fold"+str(folds)+".npy")
        # train_dataset=ECGDataset_all(train_data_path,train_label_path)
        # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8)

        # output_list=[]
        # labels_list=[]
        # for _, (data, labels) in enumerate(tqdm(train_loader)):
        #         output_list.append(data)
        #         labels_list.append(labels)
        # y_data = np.vstack(output_list)
        # y_label = np.vstack(labels_list)
        y_data = np.load(train_data_path)
        y_label = np.load(train_label_path)
        output_list_1=[]
        output_list_2=[]
        final_label_list=[]#1 the same; 0 different
        for i, j in combinations(range(y_label.shape[0]), 2):
            output_list_1.append(i)
            output_list_2.append(j)
            if np.argmax(y_label[i])==np.argmax(y_label[j]):
                final_label_list.append(np.array([[1]]))
            else:
                final_label_list.append(np.array([[0]]))
        y_label = np.vstack(final_label_list)
        y_data1=y_data[output_list_1]
        y_data2=y_data[output_list_2]
        print(diff)
        print(y_data1.shape)
        print(y_data2.shape)
        print(y_label.shape)

        np.save(train_data_path_save_1,y_data1)
        np.save(train_data_path_save_2,y_data2)
        np.save(train_label_path_save,y_label)

def pair_from_sample(arg,basepath):
    if(arg.sample_way=="random"):
        ori_path=basepath+"_random"
        folder_generate(ori_path)
    elif(arg.sample_way=="Corr"):
        ori_path=basepath+"_Corr"
        folder_generate(ori_path)
    for i in range(10):
        pair_datagen(arg,ori_path,i)

                 
    
if __name__=="__main__":
    arg=parse_args()
    basepath="mitbih"
    random.seed(arg.seed)
    pair_from_sample(arg,basepath)
            



            