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
        else:
            print("Dir exit.")
        path=Path("./result",name+"_"+diff)
        if not path.exists():
            os.makedirs(path)
        else:
            print("Dir exit.")

def few_shot(dataloader,support_path,query_path,label_path,class_index,train_data,shot):
    query_set_list=[]
    query_set_label_list=[]
    support_set_list=[]
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        query_set_list.append(data)
        query_set_label_list.append(labels)
        batch=data.shape[0]
        class_number=labels.shape[-1]
        support_temp=[]
        for class_num in range(class_number):
            index_temp=[]
            for boots in range(shot):
                index_temp.append(random.sample(class_index[class_num], batch))
            support_temp.append(train_data[index_temp])
        support_set_list.append(np.stack(support_temp,axis=1).reshape(batch,class_number*shot,-1))
    query_set_data = np.vstack(query_set_list)
    query_set_label = np.vstack(query_set_label_list)
    support_set = np.vstack(support_set_list)
    print(query_set_data.shape)
    print(query_set_label.shape)
    print(support_set.shape)
    np.save(support_path,support_set)
    np.save(query_path,query_set_data)
    np.save(label_path,query_set_label)

#ori_path=oripath+"_"+diff
def path_ori(ori_path,train,data,folds):
        return os.path.join("./data", ori_path,train,data,ori_path+"_fold"+str(folds)+".npy")

#ori_path=oripath+"_"+diff
def path_save(basepath,train,data,query,shot,folds):
    if data=="data":
        return os.path.join("./data", basepath,train,data,basepath+"_"+query+"_"+str(shot)+"_shot"+"_fold"+str(folds)+".npy")
    else:
        return os.path.join("./data", basepath,train,data,basepath+"_"+str(shot)+"_shot"+"_fold"+str(folds)+".npy")



def few_shot_dagen(basepath,folds):
    data_diff=["1","5","10","30","50","90","150"]     
    for diff in data_diff:
        ori_path=os.path.join(basepath+"_"+diff)
        test_data_path=path_ori(ori_path,"test","data",folds)
        test_label_path=path_ori(ori_path,"test","label",folds)
        val_data_path=path_ori(ori_path,"val","data",folds)
        val_label_path=path_ori(ori_path,"val","label",folds)
        train_data_path=path_ori(ori_path,"train","data",folds)
        train_label_path=path_ori(ori_path,"train","label",folds)
        
        base_path=os.path.join(basepath+"_"+diff+"_"+"pair")
        
        test_dataset=ECGDataset_all(test_data_path,test_label_path)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
        val_dataset=ECGDataset_all(val_data_path,val_label_path)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

        train_data=np.load(train_data_path)
        train_label=np.load(train_label_path)
        class_index={}
        for class_num in range(train_label.shape[-1]):
            class_index[class_num]=np.where(train_label[:,class_num]==1)[0].tolist()

        shots=[1,5]

        for shot in shots:
            val_data_path_support=path_save(base_path,"val","data","support",shot,folds)
            val_data_path_query=path_save(base_path,"val","data","query",shot,folds)
            val_label_path_save=path_save(base_path,"val","label","query",shot,folds)
            test_data_path_support=path_save(base_path,"test","data","support",shot,folds)
            test_data_path_query=path_save(base_path,"test","data","query",shot,folds)
            test_label_path_save=path_save(base_path,"test","label","query",shot,folds)

            few_shot(val_loader,val_data_path_support,val_data_path_query,val_label_path_save,class_index,train_data,shot)
            few_shot(test_loader,test_data_path_support,test_data_path_query,test_label_path_save,class_index,train_data,shot) 

def few_shot_sample(arg,basepath):
    if arg.sample_way=="random":
        basepath=basepath+"_random"
    elif arg.sample_way=="Corr":
        basepath=basepath+"_Corr"
    for i in range(10):
        few_shot_dagen(basepath,i)


    
if __name__=="__main__":
    arg=parse_args()
    datadir=arg.data_dir
    basepath="mitbih"
    random.seed(arg.seed)
    few_shot_sample(arg,basepath)
   


            



            