# -*- coding = utf-8 -*-
# @Time :2024/7/7 21:41
# @Author :郎皓东
# @File ：bpn.py
# @Software:PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,LeavePOut
from sklearn.model_selection import KFold #k折交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer,r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.linear_model as LM
from rdkit.Chem import Draw
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
import re
from collections import defaultdict

import scipy
import hyperopt
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.early_stop import no_progress_loss
import shap
import warnings
warnings.filterwarnings("ignore")
pesticide_data = pd.read_excel("../data/data_final.xlsx",sheet_name="pesticide")
PPCPs_data = pd.read_excel("../data/data_final.xlsx",sheet_name="PPCPs")
other_data = pd.read_excel("../data/data_final.xlsx",sheet_name="other")

inputX_pesticide=pesticide_data.loc[:,["log RCF","MW（g/mol）","logKow","flipid"]]
inputX_PPCPs=PPCPs_data.loc[:,["log RCF","MW（g/mol）","logKow","flipid"]]
inputX_other=other_data.loc[:,["log RCF","MW（g/mol）","logKow","flipid"]]
inputX=pd.concat([inputX_pesticide,inputX_PPCPs,inputX_other],ignore_index=True)


flipid_pesticide=pesticide_data.loc[:,"flipid"].to_numpy().reshape(-1,1)
flipid_PPCPs=PPCPs_data.loc[:,"flipid"].to_numpy().reshape(-1,1)
flipid_other=other_data.loc[:,"flipid"].to_numpy().reshape(-1,1)
flipid=np.concatenate((flipid_pesticide,flipid_PPCPs,flipid_other),0)

log_RCF_pesticide=pesticide_data.loc[:,"log RCF"].to_numpy().reshape(-1,1)
log_RCF_PPCPs=PPCPs_data.loc[:,"log RCF"].to_numpy().reshape(-1,1)
log_RCF_other=other_data.loc[:,"log RCF"].to_numpy().reshape(-1,1)
log_RCF=np.concatenate((log_RCF_pesticide,log_RCF_PPCPs,log_RCF_other),0)

MW_pesticide=pesticide_data.loc[:,"MW（g/mol）"].to_numpy().reshape(-1,1)
MW_PPCPs=PPCPs_data.loc[:,"MW（g/mol）"].to_numpy().reshape(-1,1)
MW_other=other_data.loc[:,"MW（g/mol）"].to_numpy().reshape(-1,1)
MW=np.concatenate((MW_pesticide,MW_PPCPs,MW_other),0)

logKow_pesticide=pesticide_data.loc[:,"logKow"].to_numpy().reshape(-1,1)
logKow_PPCPs=PPCPs_data.loc[:,"flipid"].to_numpy().reshape(-1,1)
logKow_other=other_data.loc[:,"flipid"].to_numpy().reshape(-1,1)
logKow=np.concatenate((logKow_pesticide,logKow_PPCPs,logKow_other),0)

SMILES_pesticide=pesticide_data.loc[:,"SMILES"]
SMILES_PPCPs=PPCPs_data.loc[:,"SMILES"]
SMILES_other=other_data.loc[:,"SMILES"]
SMILES=np.concatenate((SMILES_pesticide,SMILES_PPCPs,SMILES_other),0)

FP=[]
for SMILE in SMILES:
    info = {}
    mol=Chem.MolFromSmiles(SMILE)
    fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,bitInfo=info)
    FP.append(fp)
FP=np.array(FP)
# inputDataX_for_importance = FP
inputDataX_scaler = np.concatenate((FP,log_RCF,flipid),1)
inputDataX_scaler = scipy.stats.mstats.zscore(inputDataX_scaler)
inputDataX_for_importance = inputDataX_scaler
inputDataX_scaler=scipy.stats.mstats.zscore(inputDataX_for_importance)
inputDataX_scaler[np.isnan(inputDataX_scaler)]=0
outputDatay=np.concatenate((pesticide_data.loc[:,"log TF"],PPCPs_data.loc[:,"log TF"],other_data.loc[:,"log TF"]),0).reshape(-1,1)

def creat_chemgrp():
    input = np.c_[inputDataX_scaler, outputDatay]
    modelK=KMeans(n_clusters=3)
    modelK.fit(input)
    label_pred=modelK.labels_
    seq_row = np.arange(225).reshape(-1,1)
    input = np.c_[seq_row,input]
    x0=input[label_pred==0]
    x1=input[label_pred==1]
    x2=input[label_pred==2]
    chemical_group = []
    n_sample = len(outputDatay)
    total_id = np.arange(n_sample)
    np.random.shuffle(total_id)
    np.save("../data/sample_index.npy", total_id)
    total_id = np.load("../data/sample_index.npy")
    first_group=[]
    sec_group=[]
    third_group=[]
    for i in x0[:,0]:
        first_group.append(i)
    for i in x1[:,0]:
        sec_group.append(i)
    for i in x1[:,0]:
        third_group.append(i)
    first_group = [int(number) for number in first_group]
    sec_group = [int(number) for number in sec_group]
    third_group = [int(number) for number in third_group]


    for i in total_id:
        if i in first_group:
            chemical_group.append(0)
        elif i in sec_group:
            chemical_group.append(1)
        else:
            chemical_group.append(2)
    chemical_group = np.array(chemical_group)
    np.save("../data/chemical_group.npy", chemical_group)


