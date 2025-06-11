# -*- coding = utf-8 -*-
# @Time :2024/9/15 17:20
# @Author :郎皓东
# @File ：SVR.py
# @Software:PyCharm
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,LeavePOut
from sklearn.model_selection import KFold #k折交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer,r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import sklearn.linear_model as LM
from rdkit.Chem import Draw
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
import re
from collections import defaultdict
from utility import Kfold
from utility import MAE
import scipy
import warnings
warnings.filterwarnings("ignore")
pesticide_data = pd.read_excel("data_final.xlsx",sheet_name="pesticide")
PPCPs_data = pd.read_excel("data_final.xlsx",sheet_name="PPCPs")
other_data = pd.read_excel("data_final.xlsx",sheet_name="other")

x_pesticide_data = pesticide_data.loc[:,["log RCF","flipid"]].to_numpy().reshape(-1,2)
x_PPCPs_data=PPCPs_data.loc[:,["log RCF","flipid"]].to_numpy().reshape(-1,2)
x_other_data=other_data.loc[:,["log RCF","flipid"]].to_numpy().reshape(-1,2)
#np.random.shuffle(total_id)


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

# inputDataX_for_importance=FP
inputDataX_for_importance=np.concatenate((x_pesticide_data,x_PPCPs_data,x_other_data),0)
inputDataX_for_importance = np.concatenate((FP,inputDataX_for_importance),1)
inputDataX_scaler=scipy.stats.mstats.zscore(inputDataX_for_importance)
inputDataX_scaler[np.isnan(inputDataX_scaler)]=0
outputDatay=np.concatenate((pesticide_data.loc[:,"log TF"],PPCPs_data.loc[:,"log TF"],other_data.loc[:,"log TF"]),0).reshape(-1,1)

train_split_index,test_split_index = Kfold(len(inputDataX_for_importance),5)
total_id=np.arange(len(inputDataX_for_importance))
np.random.shuffle(total_id)

splits = 5
prediction_svc = []
prediction_true_svc = []
test_score_all_svc = []
for k in range(splits):

    print('batch is ',k)
    train_index = train_split_index[k][:int(len(train_split_index[k])*0.875)]
    valid_index = train_split_index[k][int(len(train_split_index[k])*0.875):]
    test_index = test_split_index[k]

    train_id = [total_id[i] for i in train_index]
    valid_id = [total_id[i] for i in valid_index]
    test_id = [total_id[i] for i in test_index]

    train_feature = [inputDataX_scaler[i] for i in train_id]
    train_label = [outputDatay[i] for i in train_id]

    valid_feature = [inputDataX_scaler[i] for i in valid_id]
    valid_label = [outputDatay[i] for i in valid_id]

    test_feature = [inputDataX_scaler[i] for i in test_id]
    test_label = [outputDatay[i] for i in test_id]

    G_pool = [0.00001,0.0001,0.001, 0.01, 0.1, 1,10,100]

    C_pool = [0.0001,0.001, 0.01, 0.1, 1, 10,25,50,100,1000]

    best_valid_score = float('-inf')
    for c in C_pool:
        for g in G_pool:
            model = SVR(kernel='rbf',C=c,gamma=g)
            model.fit(train_feature,train_label)
            valid_score = model.score(valid_feature,valid_label)
            #print('valid score is',valid_score)
            if valid_score>best_valid_score:
                best_valid_score = valid_score
                test_score = model.score(test_feature,np.array(test_label).reshape(-1,1))
                best_n = c
                best_d = g
                pred = model.predict(test_feature)
    print('test score is',test_score)
    test_score_all_svc.append(test_score)
    prediction_svc.append(pred)
    prediction_true_svc.append(test_label)
    print('best c is',best_n)
    print('best g is',best_d)
    #print('feature importance',model.feature_importances_)


model = SVR(kernel='rbf',C=best_n,gamma=best_d)
prediction_svc_all = []
for l in prediction_svc:
    for v in l:
        prediction_svc_all.append(v)



prediction_true_svc_all = []
for l in prediction_true_svc:
    for v in l:
        prediction_true_svc_all.append(v)


prediction_true_svc_all = np.array(prediction_true_svc_all).flatten()
prediction_svc_all = np.array(prediction_svc_all)

fig = plt.figure(figsize=(8,6))
ax0 = fig.add_subplot(111)
sns.scatterplot(x=prediction_true_svc_all,y=prediction_svc_all)
sns.lineplot(x=np.arange(-3,3.),y=np.arange(-3,3.),color='r')
r_2=r2_score(prediction_true_svc_all,prediction_svc_all)
# MAE=np.abs(cross_val_score(model,prediction_true_svc_all,prediction_svc_all,cv=5,scoring='neg_mean_absolute_error')).mean()
MAE = MAE(prediction_true_svc_all,prediction_svc_all)
ax0.text(0.05, 0.95,f"R-squared={round(r_2,2)} MAE={round(MAE,2)}", transform=ax0.transAxes, ha='left', va='top')
plt.xlabel('Measured logRCF',fontsize=14)
plt.ylabel('Predicted logRCF',fontsize=14)

plt.show()
