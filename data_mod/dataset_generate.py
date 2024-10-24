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
from data_mod.dataset import ECGDataset
from sklearn.model_selection import train_test_split, KFold
import random


def folder_generate(name):
    data_diff=["1","5","10","30","50","90","150"]

    for diff in data_diff:
        path=Path("./data",name+"_"+diff)
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

def Corr_sample_datagen(path_dic,basepath,fold):
    data_diff=["1","5","10","30","50","90","150"]
    X_train = np.load(path_dic["train_data"])
    y_train=np.load(path_dic["train_label"])
    ind_train = np.load(path_dic["corr_index"])
    y_train=np.load(path_dic["train_label"])
    X_test=np.load(path_dic["test_data"])
    y_test=np.load(path_dic["test_label"])
    X_val=np.load(path_dic["val_data"])
    y_val = np.load(path_dic["val_label"])
    for diff in data_diff:
        train_data_path=os.path.join("./data",basepath+"_"+diff,"train","data",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        train_label_path=os.path.join("./data",basepath+"_"+diff,"train","label",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        val_data_path=os.path.join("./data",basepath+"_"+diff,"val","data",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        val_label_path=os.path.join("./data",basepath+"_"+diff,"val","label",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        test_data_path=os.path.join("./data",basepath+"_"+diff,"test","data",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        test_label_path=os.path.join("./data",basepath+"_"+diff,"test","label",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        random_num=int(diff)
        train_data=[]
        train_label=[]
        label_class_data=np.argmax(y_train,1)
        for class_num in range(y_train.shape[-1]):
            c_index=np.where(label_class_data==class_num)[0]
            c_ind_train=ind_train[c_index]
            sorted_c_ind_train=np.argsort(c_ind_train)
            sorted_c_ind_train = sorted_c_ind_train[::-1]
            shot_temp= c_index[sorted_c_ind_train[:random_num]]
            train_data=train_data+shot_temp
            train_label=train_label+shot_temp
        data=X_train[train_data]
        label=y_train[train_label]
        print(diff)
        print(data.shape)
        print(label.shape)
        np.save(train_data_path,data)
        np.save(train_label_path,label)
        print(diff)
        print(X_val.shape)
        print(y_val.shape)
        np.save(val_data_path,X_val)
        np.save(val_label_path,y_val)
        print(diff)
        print(X_test.shape)
        print(y_test.shape)
        np.save(test_data_path,X_test)
        np.save(test_label_path,y_test)

def random_sample_datagen(path_dic,basepath,fold):
    data_diff=["1","5","10","30","50","90","150"]
    X_train = np.load(path_dic["train_data"])
    y_train=np.load(path_dic["train_label"])
    X_test=np.load(path_dic["test_data"])
    y_test=np.load(path_dic["test_label"])
    X_val=np.load(path_dic["val_data"])
    y_val = np.load(path_dic["val_label"])

    for diff in data_diff:
        train_data_path=os.path.join("./data",basepath+"_"+diff,"train","data",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        train_label_path=os.path.join("./data",basepath+"_"+diff,"train","label",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        val_data_path=os.path.join("./data",basepath+"_"+diff,"val","data",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        val_label_path=os.path.join("./data",basepath+"_"+diff,"val","label",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        test_data_path=os.path.join("./data",basepath+"_"+diff,"test","data",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        test_label_path=os.path.join("./data",basepath+"_"+diff,"test","label",basepath+"_"+diff+"_fold"+str(fold)+".npy")
        random_num=int(diff)
        train_data=[]
        train_label=[]
        for class_num in range(y_train.shape[-1]):
            c_index=frozenset(np.where(y_train[:,class_num]==1)[0].tolist())
            shot_temp= random.sample(c_index, random_num)
            train_data=train_data+shot_temp
            train_label=train_label+shot_temp
        data=X_train[train_data]
        label=y_train[train_label]
        print(diff)
        print(data.shape)
        print(label.shape)
        np.save(train_data_path,data)
        np.save(train_label_path,label)
        print(diff)
        print(X_val.shape)
        print(y_val.shape)
        np.save(val_data_path,X_val)
        np.save(val_label_path,y_val)
        print(diff)
        print(X_test.shape)
        print(y_test.shape)
        np.save(test_data_path,X_test)
        np.save(test_label_path,y_test)

def US_from_all(arg,basepath):
    if(arg.sample_way=="random"):
            basepath=basepath+"_random"
            folder_generate(basepath)
    elif(arg.sample_way=="Corr"):
            basepath=basepath+"_Corr"
            folder_generate(basepath)
    for i in range(10):
        
        path_dic={}
        path_dic["train_data"]=os.path.join("./data","mitbih_all","train","data","mitbih_all"+"_fold"+str(i)+".npy")
        path_dic["train_label"]=os.path.join("./data","mitbih_all","train","label","mitbih_all"+"_fold"+str(i)+".npy")
        path_dic["train_corr_index"]=os.path.join("./data","mitbih_all","train","corr_index","mitbih_all"+"_fold"+str(i)+".npy")
        path_dic["val_data"]=os.path.join("./data","mitbih_all","val","data","mitbih_all"+"_fold"+str(i)+".npy")
        path_dic["val_label"]=os.path.join("./data","mitbih_all","val","label","mitbih_all"+"_fold"+str(i)+".npy")
        path_dic["test_data"]=os.path.join("./data","mitbih_all","test","data","mitbih_all"+"_fold"+str(i)+".npy")
        path_dic["test_label"]=os.path.join("./data","mitbih_all","test","label","mitbih_all"+"_fold"+str(i)+".npy")

        if(arg.sample_way=="random"):
            random_sample_datagen(path_dic,basepath,i)
    
        elif(arg.sample_way=="Corr"):
            random_sample_datagen(path_dic,basepath,i)



    
if __name__=="__main__":
    arg=parse_args()
    basepath="mitbih"
    random.seed(arg.seed)
    US_from_all(arg,basepath)
   
   
            



            