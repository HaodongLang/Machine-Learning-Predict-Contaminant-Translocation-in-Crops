# -*- coding = utf-8 -*-
# @Time :2024/10/8 10:23
# @Author :郎皓东
# @File ：data_boxplot.py
# @Software:PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
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
from utility import Kfold
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

x_pesticide_RCF = pesticide_data.loc[:,["log RCF"]].to_numpy().reshape(-1,1)
x_PPCPs_RCF = PPCPs_data.loc[:,["log RCF"]].to_numpy().reshape(-1,1)
x_other_RCF = other_data.loc[:,["log RCF"]].to_numpy().reshape(-1,1)
plt_RCF = np.concatenate((x_pesticide_RCF,x_PPCPs_RCF,x_other_RCF),0)
# RCF_row = np.full(plt_RCF.shape[0],1)
# plt_RCF = np.c_[plt_RCF,RCF_row]

x_pesticide_flipid = pesticide_data.loc[:,["flipid"]].to_numpy().reshape(-1,1)
x_PPCPs_flipid = PPCPs_data.loc[:,["flipid"]].to_numpy().reshape(-1,1)
x_other_flipid = other_data.loc[:,["flipid"]].to_numpy().reshape(-1,1)
plt_flipid = np.concatenate((x_pesticide_flipid,x_PPCPs_flipid,x_other_flipid),0)
# flipid_row = np.full(plt_flipid.shape[0],1)
# plt_flipid = np.c_[plt_flipid,flipid_row]


x_pesticide_MW = pesticide_data.loc[:,["MW（g/mol）"]].to_numpy().reshape(-1,1)
x_PPCPs_MW = PPCPs_data.loc[:,["MW（g/mol）"]].to_numpy().reshape(-1,1)
x_other_MW = other_data.loc[:,["MW（g/mol）"]].to_numpy().reshape(-1,1)
plt_MW = np.concatenate((x_pesticide_MW,x_PPCPs_MW,x_other_MW),0)
# MW_row = np.full(plt_MW.shape[0],1)
# plt_MW = np.c_[plt_MW,MW_row]

x_pesticide_logKow = pesticide_data.loc[:,["logKow"]].to_numpy().reshape(-1,1)
x_PPCPs_logKow = PPCPs_data.loc[:,["logKow"]].to_numpy().reshape(-1,1)
x_other_logKow = other_data.loc[:,["logKow"]].to_numpy().reshape(-1,1)
plt_logKow = np.concatenate((x_pesticide_logKow,x_PPCPs_logKow,x_other_logKow),0)
# logKow_row = np.full(plt_logKow.shape[0],1)
# plt_logKow = np.c_[plt_logKow,logKow_row]

outputDatay=np.concatenate((pesticide_data.loc[:,"log TF"],PPCPs_data.loc[:,"log TF"],other_data.loc[:,"log TF"]),0).reshape(-1,1)
TF_row = np.full(outputDatay.shape[0],1)
plt_TF = np.c_[outputDatay,TF_row]



FP=[]
MW_unique = []
logKow_unqiue = []
flipid_unique = []
SMILES_pesticide=pesticide_data.loc[:,"SMILES"]
SMILES_PPCPs=PPCPs_data.loc[:,"SMILES"]
SMILES_other=other_data.loc[:,"SMILES"]
SMILES=np.concatenate((SMILES_pesticide,SMILES_PPCPs,SMILES_other),0)
for i,sm in enumerate(SMILES):
    mol = Chem.MolFromSmiles(sm)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    if np.array(fp).tolist() not in FP:
        FP.append(np.array(fp).tolist())
        MW_unique.append(plt_MW[i])
        logKow_unqiue.append(plt_logKow[i])

plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
plt.rcParams['font.weight'] = 'bold'   # 设置字体加粗
# 设置全局字体属性
font = FontProperties()
font.set_family('serif')
# font.set_weight('bold')
font.set_name('Times New Roman')
# 设置全局字体样式
plt.rc('font', family='Times New Roman')
figs,new_ax = plt.subplots(2,3,figsize=(12,10))
plt.subplots_adjust(wspace=0.5)
for i in range(3):
    for spine in ['top', 'left', 'right', 'bottom']:
        new_ax[0][i].spines[spine].set_linewidth(1.5)  # 将边框宽度设置为2
for i in range(2):
    for spine in ['top', 'left', 'right', 'bottom']:
        new_ax[1][i].spines[spine].set_linewidth(1.5)  # 将边框宽度设置为2

plt.yticks(weight='bold')

sns.swarmplot(plt_RCF[:,0],ax=new_ax[0][0],orient='v',s=3.5,color="red")
# new_ax[0][0].scatter(plt_RCF[:,1],plt_RCF[:,0],c="red",zorder=1)
sns.boxplot(plt_RCF[:,0],ax=new_ax[0][0],orient='v',color='pink',linewidth=3)
new_ax[0][0].set_xlabel("logRCF",fontsize=30,fontproperties=font)
new_ax[0][0].tick_params(axis='y', labelsize=30)
new_ax[0][0].xaxis.set_major_locator(plt.NullLocator())
for tick in new_ax[0][0].yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

bold_text = r'f{lipid}'
sns.swarmplot(list(set(plt_flipid.reshape(-1))),ax=new_ax[0][1],orient='v',s=3.5,color="g")
sns.boxplot(plt_flipid[:,0],ax=new_ax[0][1],orient='v',color='#90EE90',linewidth=3)
new_ax[0][1].set_xlabel(r"f$_{lipid}$",fontsize=30,fontproperties=font)
new_ax[0][1].tick_params(axis='y', labelsize=30)
new_ax[0][1].xaxis.set_major_locator(plt.NullLocator())
for tick in new_ax[0][1].yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

sns.swarmplot(np.array(MW_unique).reshape(-1),ax=new_ax[0][2],orient='v',s=3.5,color="y")
sns.boxplot(plt_MW[:,0],ax=new_ax[0][2],orient='v',color='#FFFFE0',linewidth=3)
new_ax[0][2].set_xlabel("MW",fontsize=30,fontproperties=font)
new_ax[0][2].tick_params(axis='y', labelsize=30)
new_ax[0][2].xaxis.set_major_locator(plt.NullLocator())
for tick in new_ax[0][2].yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

sns.swarmplot(np.array(logKow_unqiue).reshape(-1),ax=new_ax[1][0],orient='v',s=3.5,color="blue")
sns.boxplot(plt_logKow[:,0],ax=new_ax[1][0],orient='v',color='#E6E6FA',linewidth=3)
new_ax[1][0].set_xlabel("logK$_{ow}$",fontsize=30,fontproperties=font)
new_ax[1][0].tick_params(axis='y', labelsize=30)
new_ax[1][0].xaxis.set_major_locator(plt.NullLocator())
for tick in new_ax[1][0].yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

sns.swarmplot(plt_TF[:,0],ax=new_ax[1][1],orient='v',s=3.5,color="y")
sns.boxplot(plt_TF[:,0],ax=new_ax[1][1],orient='v',color='#FFFFE0',linewidth=3)
new_ax[1][1].set_xlabel("logTF",fontsize=30,fontproperties=font)
new_ax[1][1].tick_params(axis='y', labelsize=30)
for tick in new_ax[1][1].yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
plt.savefig('boxplot.jpg')
plt.show()


