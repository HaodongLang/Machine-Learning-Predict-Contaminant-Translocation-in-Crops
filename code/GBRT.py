# -*- coding = utf-8 -*-
# @Time :2024/7/11 16:11
# @Author :郎皓东
# @File ：GBRT&ECFP.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @Time :2024/1/17 0:10
# @Author :郎皓东
# @File ：ECFP.py
# @Software:PyCharm
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split,LeavePOut
from sklearn.model_selection import KFold #k折交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import plot_partial_dependence
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
import seaborn as sns
import re
from collections import defaultdict
from utility import Kfold
import scipy
import hyperopt
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.early_stop import no_progress_loss
import shap
import matplotlib.font_manager as fm
import os
import warnings
warnings.filterwarnings("ignore")
#load data

pesticide_data = pd.read_excel("../data/data_final.xlsx",sheet_name="pesticide")
PPCPs_data = pd.read_excel("../data/data_final.xlsx",sheet_name="PPCPs")
other_data = pd.read_excel("../data/data_final.xlsx",sheet_name="other")

name_condition_α = other_data['Compound'] == "α-HBCD"
name_condition_b = other_data['Compound'] == "β-HBCD"

α_HBCD = other_data[name_condition_α]
FP=[]
for SMILE in α_HBCD.loc[:,"SMILES"]:
    info = {}
    mol=Chem.MolFromSmiles(SMILE)
    fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,bitInfo=info)
    FP.append(fp)
FP = np.array(FP)
α_HBCD = α_HBCD.loc[:,["log RCF","MW（g/mol）","logKow","flipid"]]
b_HBCD = other_data[name_condition_b]

ECFP_α_HBCD = np.concatenate((α_HBCD.loc[:,["log RCF","flipid"]],FP),1)
FP=[]
for SMILE in b_HBCD.loc[:,"SMILES"]:
    info = {}
    mol=Chem.MolFromSmiles(SMILE)
    fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,bitInfo=info)
    FP.append(fp)
FP = np.array(FP)
b_HBCD = b_HBCD.loc[:,["log RCF","MW（g/mol）","logKow","flipid"]]
ECFP_b_HBCD = np.concatenate((b_HBCD.loc[:,["log RCF","flipid"]],FP),1)


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
logKow_PPCPs=PPCPs_data.loc[:,"logKow"].to_numpy().reshape(-1,1)
logKow_other=other_data.loc[:,"logKow"].to_numpy().reshape(-1,1)
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
inputDataX_scaler = np.concatenate((log_RCF,flipid,MW,logKow),1)
# inputDataX_scaler=scipy.stats.mstats.zscore(inputDataX_scaler)

inputDataX_scaler[np.isnan(inputDataX_scaler)]=0
outputDatay=np.concatenate((pesticide_data.loc[:,"log TF"],PPCPs_data.loc[:,"log TF"],other_data.loc[:,"log TF"]),0).reshape(-1,1)

train_split_index,test_split_index=Kfold(len(inputDataX_scaler),5)
total_id=np.load("../data/sample_index.npy")
feature_importance_all_w_smiles=[]
permute_importance_all_w_smiles_test=[]
test_score_all = []
prediction_w_smiles = []
prediction_true_w_smiles = []
test_score_all_w_smiles = []


#贝叶斯优化
def opt_object(params):
    score = []
    model = GradientBoostingRegressor(n_estimators = int(params["n_estimators"])
              ,max_depth = int(params["max_depth"])
              ,learning_rate = params["lr"]
              # ,min_samples_split=int(params["min_samples_split"])
              ,max_features = int(params["max_features"])
              ,subsample = params["subsample"]
              # ,random_state = 1412
              # ,verbose = False
              )

    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    validation_loss = cross_validate(model, inputDataX_scaler, np.ravel(outputDatay)
                                     , scoring="r2"
                                     , cv=cv
                                     , verbose=False
                                     , n_jobs=-1
                                     , error_score='raise'
                                     )
    return np.mean(abs(validation_loss["test_score"]))
    #return round(np.mean(score),2)
param_grid_simple = {'n_estimators': hp.quniform("n_estimators",200,1250,50)
                  ,"lr": hp.quniform("learning_rate",0.1,0.2,0.01)
                  ,"max_depth": hp.quniform("max_depth",2,10,2)
                  ,"subsample": hp.quniform("subsample",0.5,1,0.1)
                  ,"max_features": hp.quniform("max_features",1,4,1)
                 }


def param_hyperopt(max_evals=100):
    # 保存迭代过程
    trials = Trials()

    # 设置提前停止
    early_stop_fn = no_progress_loss(300)

    # 定义代理模型
    params_best = fmin(opt_object
                       , space=param_grid_simple
                       , algo=tpe.suggest
                       , max_evals=max_evals
                       , verbose=True
                       , trials=trials
                       , early_stop_fn=early_stop_fn
                       )

    # 打印最优参数，fmin会自动打印最佳分数
    print("\n", "\n", "best params: ", params_best,
          "\n")
    return params_best, trials
params_best, trials = param_hyperopt(1)
print(params_best,trials)


best_score=0
prediction_gbrt = []
prediction_true_gbrt = []
fig = []


for k in range(5):
    train_index = train_split_index[k][:int(len(train_split_index[k]) * 0.875)]
    val_index = train_split_index[k][int(len(train_split_index[k]) * 0.875):]
    test_index = test_split_index[k]
    train_id = []
    val_id = []
    test_id = []
    for i in train_index:
        train_id.append(total_id[i])
    for i in val_index:
        val_id.append(total_id[i])
    for i in test_index:
        test_id.append(total_id[i])

    train_feature = [inputDataX_scaler[i] for i in train_id]
    train_label = [outputDatay[i] for i in train_id]

    val_feature = [inputDataX_scaler[i] for i in val_id]
    val_label = [outputDatay[i] for i in val_id]

    test_feature = [inputDataX_scaler[i] for i in test_id]
    test_label = [outputDatay[i] for i in test_id]

    model=GradientBoostingRegressor(n_estimators=int(params_best['n_estimators']),max_depth=int(params_best['max_depth']),learning_rate=params_best['learning_rate'],max_features=int(params_best['max_features']),subsample= params_best['subsample'])
    # model = GradientBoostingRegressor(n_estimators=int(params_best['n_estimators']),max_depth=int(params_best['max_depth']))
    model.fit(X=np.array(train_feature),y=np.array(train_label).reshape(-1,1))
    pred = model.predict(test_feature)

    train_score = model.score(np.array(train_feature), np.array(train_label).reshape(-1, 1))
    val_score = model.score(np.array(val_feature), np.array(val_label).reshape(-1, 1))
    test_score = model.score(np.array(test_feature), np.array(test_label).reshape(-1, 1))

    permut_importance_test=permutation_importance(model,np.array(test_feature),np.array(test_label).reshape(-1,1),n_repeats=10)
    feature_importance_all_w_smiles.append(model.feature_importances_)      #all five splits feature importances
    permute_importance_all_w_smiles_test.append(permut_importance_test.importances_mean)
    fig.append(plot_partial_dependence(model,np.array(train_feature),[0,1,2,3],feature_names=["log RCF","flipid","MW","logKow"],method = 'brute',kind = 'both',percentiles=(0,1)))
    prediction_gbrt.append(pred)
    prediction_true_gbrt.append(test_label)
    test_score_all.append(test_score)

    print("The train score for this split is",train_score,"The val score for this split is",val_score,"The test score for this split is",test_score)


print(model.predict(α_HBCD))
print(model.predict(b_HBCD))
mean_feature_importance_all_impurity_smiles=np.mean(feature_importance_all_w_smiles,0)
sorted_feature_imporatnce_idx_impurity_smiles=np.argsort(mean_feature_importance_all_impurity_smiles[:-2])[::-1]
top_10_impurity=sorted_feature_imporatnce_idx_impurity_smiles[:10]
print('The top 10 most important substructure id from impurity importance is',top_10_impurity)

mean_feature_importance_all_permute_smiles_test=np.mean(permute_importance_all_w_smiles_test,0)
sorted_feature_imporatnce_idx_permute_smiles_test=np.argsort(mean_feature_importance_all_permute_smiles_test[:-2])[::-1]
top_10_permute=sorted_feature_imporatnce_idx_permute_smiles_test[:10]
print('The top 10 most important substructure id from permutation importance is',top_10_permute)
print("mean_feature_importance_all_permute_smiles_test is",mean_feature_importance_all_permute_smiles_test)

plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
plt.rcParams['font.weight'] = 'bold'   # 设置字体加粗
plt.rcParams['font.serif'] = ['Times New Roman'] # 设置全局字体样式

figs,new_ax = plt.subplots(2,2,figsize=(18,18))

fig_ = plt.figure(figsize=(8,6))
ax0 = fig_.add_subplot(111)

# 设置全局字体属性
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size('25')
font.set_weight('bold')
# 设置全局字体样式
plt.rc('font', family='Times New Roman')

for i in range(157):
    new_ax[0][0].plot(fig[2].pd_results[0]['values'][0],fig[2].pd_results[0]['individual'][0][i],color='salmon',alpha=0.05)
new_ax[0][0].plot(fig[2].pd_results[0]['values'][0],fig[2].pd_results[0]['average'][0],color='red',label='average')
new_ax[0][0].tick_params(axis="both",labelsize=35,width=3)
new_ax[0][0].set_ylabel('Partial Dependence',fontsize=30,fontproperties=font)
new_ax[0][0].set_xlabel('logRCF',fontsize=30,fontproperties=font)
# 加粗x轴
new_ax[0][0].spines['bottom'].set_linewidth(3)  # 底部x轴
new_ax[0][0].spines['top'].set_linewidth(3)  # 顶部x轴（如果需要）
# 加粗y轴
new_ax[0][0].spines['left'].set_linewidth(3)  # 左侧y轴
new_ax[0][0].spines['right'].set_linewidth(3)  # 右侧y轴（如果需要）
legend = new_ax[0][0].legend(loc="upper right",prop=font)
frame = legend.get_frame()
frame.set_edgecolor('black')  # 设置边框颜色
frame.set_linewidth(2.0)     # 设置边框线宽
for i in range(157):
    new_ax[0][1].plot(fig[2].pd_results[1]['values'][0],fig[2].pd_results[1]['individual'][0][i],color='yellowgreen',alpha=0.05)
new_ax[0][1].plot(fig[2].pd_results[1]['values'][0],fig[2].pd_results[1]['average'][0],color='green',label='average')
new_ax[0][1].tick_params(axis="both",labelsize=35,width=3)
new_ax[0][1].set_ylabel('Partial Dependence',fontsize=30,fontproperties=font)
new_ax[0][1].set_xlabel(r'f$_{lipid}$',fontsize=30,fontproperties=font)
# 加粗x轴
new_ax[0][1].spines['bottom'].set_linewidth(3)  # 底部x轴
new_ax[0][1].spines['top'].set_linewidth(3)  # 顶部x轴（如果需要）
# 加粗y轴
new_ax[0][1].spines['left'].set_linewidth(3)  # 左侧y轴
new_ax[0][1].spines['right'].set_linewidth(3)  # 右侧y轴（如果需要）
legend = new_ax[0][1].legend(loc="upper right",prop=font)
frame = legend.get_frame()
frame.set_edgecolor('black')  # 设置边框颜色
frame.set_linewidth(2.0)     # 设置边框线宽
for i in range(157):
    new_ax[1][0].plot(fig[2].pd_results[2]['values'][0],fig[2].pd_results[2]['individual'][0][i],color='khaki',alpha=0.05)
new_ax[1][0].plot(fig[2].pd_results[2]['values'][0],fig[2].pd_results[2]['average'][0],color='gold',label='average')
new_ax[1][0].tick_params(axis="both",labelsize=35,width=3)
new_ax[1][0].set_ylabel('Partial Dependence',fontsize=30,fontproperties=font)
new_ax[1][0].set_xlabel('MW(g/mol)',fontsize=30,fontproperties=font)
# 加粗x轴
new_ax[1][0].spines['bottom'].set_linewidth(3)  # 底部x轴
new_ax[1][0].spines['top'].set_linewidth(3)  # 顶部x轴（如果需要）
# 加粗y轴
new_ax[1][0].spines['left'].set_linewidth(3)  # 左侧y轴
new_ax[1][0].spines['right'].set_linewidth(3)  # 右侧y轴（如果需要）
legend = new_ax[1][0].legend(loc="upper right",prop=font)
frame = legend.get_frame()
frame.set_edgecolor('black')  # 设置边框颜色
frame.set_linewidth(2.0)     # 设置边框线宽
for i in range(157):
    new_ax[1][1].plot(fig[2].pd_results[3]['values'][0],fig[2].pd_results[3]['individual'][0][i],color='lightskyblue',alpha=0.05)
new_ax[1][1].plot(fig[2].pd_results[3]['values'][0],fig[2].pd_results[3]['average'][0],color='deepskyblue',label='average')
new_ax[1][1].tick_params(axis="both",labelsize=35,width=3)
new_ax[1][1].set_ylabel('Partial Dependence',fontsize=30,fontproperties=font)
new_ax[1][1].set_xlabel('logK$_{ow}$',fontsize=30,fontproperties=font)
# 加粗x轴
new_ax[1][1].spines['bottom'].set_linewidth(3)  # 底部x轴
new_ax[1][1].spines['top'].set_linewidth(3)  # 顶部x轴（如果需要）
# 加粗y轴
new_ax[1][1].spines['left'].set_linewidth(3)  # 左侧y轴
new_ax[1][1].spines['right'].set_linewidth(3)  # 右侧y轴（如果需要）
legend = new_ax[1][1].legend(loc="upper right",prop=font)
frame = legend.get_frame()
frame.set_edgecolor('black')  # 设置边框颜色
frame.set_linewidth(2.0)     # 设置边框线宽

ccmap = np.load('../chemical_group.npy')

ccmap_shuffle_property = []
for i in total_id:
    ccmap_shuffle_property.append(ccmap[i])

prediction_gbrt_all = []
for l in prediction_gbrt:
    for v in l:
        prediction_gbrt_all.append(v)

prediction_true_gbrt_all = []
for l in prediction_true_gbrt:
    for v in l:
        prediction_true_gbrt_all.append(v)

# 设置全局字体样式
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
plt.rcParams['font.weight'] = 'bold'   # 设置字体加粗
plt.rcParams['font.serif'] = ['Times New Roman'] # 设置全局字体样式

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)
ax.tick_params(axis='both', labelsize=35,width=3)
# 设置全局字体属性
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size('18')


plt.xlabel('measured logTF',fontsize=30,fontproperties=font)
plt.ylabel('predicted logTF',fontsize=30,fontproperties=font)
sns.scatterplot(x=np.array(prediction_true_gbrt_all).reshape(-1).tolist(),y=prediction_gbrt_all,linewidth=0.5,hue=ccmap_shuffle_property,palette={1:'#6D042C',0:"#EECF63",2:"#3E7E6C"},s=35)
sns.lineplot(x=np.arange(-3.,3.0),y=np.arange(-3,3.0),color='r')
r_2=r2_score(prediction_true_gbrt_all,prediction_gbrt_all)
MAE=mean_absolute_error(prediction_true_gbrt_all,prediction_gbrt_all)
# 加粗x轴
ax.spines['bottom'].set_linewidth(3)  # 底部x轴
ax.spines['top'].set_linewidth(3)  # 顶部x轴（如果需要）

# 加粗y轴
ax.spines['left'].set_linewidth(3)  # 左侧y轴
ax.spines['right'].set_linewidth(3)  # 右侧y轴（如果需要）
handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles, ['A', 'B', 'C'],fontsize=30,loc="lower right")
frame = legend.get_frame()
frame.set_edgecolor('black')  # 设置边框颜色为黑色
frame.set_linewidth(2)  # 设置边框宽度为2.0
plt.text(-3,2.1,"R-squared = %0.2f MAE = %0.2f" % (np.mean(test_score_all),MAE),ha='left',va='top',fontsize=25,fontproperties=font)

plt.show()
