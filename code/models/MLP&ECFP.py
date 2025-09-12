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
import creat_chemgrp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.early_stop import no_progress_loss
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split, LeavePOut, cross_validate
from sklearn.model_selection import KFold #k折交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import sklearn.linear_model as LM
from rdkit.Chem import Draw
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
import re
from collections import defaultdict
from utility import Kfold,error
import scipy
import warnings
import time
warnings.filterwarnings("ignore")
creat_chemgrp.creat_chemgrp()


# timeout = 5400  # 设置超时时间为10秒
# start_time = time.time()
#
#
# while True:


pesticide_data = pd.read_excel("../data/data_final.xlsx",sheet_name="pesticide")
PPCPs_data = pd.read_excel("../data/data_final.xlsx",sheet_name="PPCPs")
other_data = pd.read_excel("../data/data_final.xlsx",sheet_name="other")

x_pesticide_data = pesticide_data.loc[:,["log RCF","flipid","TPSA","HBD","HBA"]].to_numpy().reshape(-1,5)
x_PPCPs_data=PPCPs_data.loc[:,["log RCF","flipid","TPSA","HBD","HBA"]].to_numpy().reshape(-1,5)
x_other_data=other_data.loc[:,["log RCF","flipid","TPSA","HBD","HBA"]].to_numpy().reshape(-1,5)

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

inputDataX_for_importance=np.concatenate((x_pesticide_data,x_PPCPs_data,x_other_data),0)
inputDataX_scaler = np.concatenate((inputDataX_for_importance,FP),1)
# inputDataX_scaler=scipy.stats.mstats.zscore(inputDataX_for_importance)
scaler = MinMaxScaler()
# inputDataX_scaler = scaler.fit_transform(inputDataX_scaler)
# inputDataX_scaler[np.isnan(inputDataX_scaler)]=0
outputDatay=np.concatenate((pesticide_data.loc[:,"log TF"],PPCPs_data.loc[:,"log TF"],other_data.loc[:,"log TF"]),0).reshape(-1,1)
immature_data = inputDataX_scaler[:3]
immature_TF = outputDatay[:3]
immature_TF = immature_TF.ravel()
#数据集中成熟作物数据读取
condition = other_data["Citation"].str.contains('''Hexabromocyclododecanes in surface soil-maize system around
Baiyangdian Lake in North China: Distribution, enantiomer-specific''')
riped_data = other_data[condition]
riped_TF = riped_data.loc[:,"log TF"]
riped_TF = np.array(riped_TF)
FP=[]
for SMILE in riped_data.loc[:,"SMILES"]:
    info = {}
    mol=Chem.MolFromSmiles(SMILE)
    fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,bitInfo=info)
    FP.append(fp)
FP = np.array(FP)
TF_data = riped_data.loc[:,"log TF"]
TF_data = np.array(TF_data)
# riped_data = np.concatenate((riped_data.loc[:,["log RCF","flipid"]],FP),1)
#数据集外成熟数据读取
data = pd.read_excel("../data/副本data_final(2)(2).xlsx",sheet_name="Sheet1")
extra_riped_data=data.tail(3).loc[:,["logRCF","lipid"]]
extra_FP=[]
for SMILE in data.tail(3).loc[:,"SMILES"]:
    info = {}
    mol=Chem.MolFromSmiles(SMILE)
    fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,bitInfo=info)
    extra_FP.append(fp)
extra_FP=np.array(extra_FP)
extra_riped_data = np.concatenate((extra_riped_data,extra_FP),1)
extra_TF_data = data.tail(3).loc[:,"logTF"]
extra_TF_data = np.array(extra_TF_data)

train_split_index,test_split_index=Kfold(len(inputDataX_for_importance),5)
# print(np.array(train_split_index)[0],"\n",np.array(train_split_index)[1])
total_id=np.load("../data/sample_index.npy")
prediction_mlp = []
prediction_true_mlp = []
importance_all_dots_nn = []
feature_importance_all_w_smiles=[]
permute_importance_all_w_smiles_test=[]
test_score_all = []

fig = []
n_estimator = [(512,256,128),(512,256),(256,128),(256,128,64)]
batch_size = [64,128]
#贝叶斯优化
def opt_object(params):
    score = []
    model = MLPRegressor(hidden_layer_sizes = params["hidden_layer_sizes"]
              ,learning_rate_init = round(params["learning_rate_init"],4)
              ,max_iter = int(params['max_iter'])
              ,batch_size = int(params["batch_size"])
              )

    cv = KFold(n_splits=5, shuffle=True,random_state=1412)
    validation_loss = cross_validate(model, inputDataX_scaler, np.ravel(outputDatay)
                                     , scoring="r2"
                                     , cv=cv
                                     , verbose=False
                                     , n_jobs=-1
                                     , error_score='raise'
                                     )
    return np.mean(abs(validation_loss["test_score"]))
    #return round(np.mean(score),2)




param_grid_simple = {'hidden_layer_sizes': hp.choice("hidden_layer_sizes",[(512,256,128),(512,256),(256,128),(256,128,64)])
                  ,"learning_rate_init": hp.quniform("learning_rate_init",0.0001,0.001,0.0001)
                  ,"max_iter":hp.quniform("max_iter",500,2000,500)
                  ,"batch_size": hp.choice("batch_size",[64,128])
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

for k in range(5):
    print("The split is",k)
    train_index = train_split_index[k][:int(len(train_split_index[k]) * 0.875)]
    val_index = train_split_index[k][int(len(train_split_index[k]) * 0.875):]
    test_index=test_split_index[k]
    train_id = []
    val_id = []
    test_id = []
    for i in train_index:
        train_id.append(total_id[i])
    for i in val_index:
        val_id.append(total_id[i])
    for i in test_index:
        test_id.append(total_id[i])

    train_feature=[inputDataX_scaler[i]for i in train_id]
    train_label=[outputDatay[i] for i in train_id]

    val_feature=[inputDataX_scaler[i] for i in val_id]
    val_label=[outputDatay[i] for i in val_id]

    test_feature=[inputDataX_scaler[i] for i in test_id]
    test_label=[outputDatay[i] for i in test_id]

    best_score = 0


    model = MLPRegressor(hidden_layer_sizes=n_estimator[params_best["hidden_layer_sizes"]]
                         ,solver="adam"
                         ,learning_rate_init=round(params_best["learning_rate_init"],4)
                         ,batch_size=batch_size[params_best["batch_size"]]
                         ,max_iter=int(params_best["max_iter"])
                         )
    model.fit(X=np.array(train_feature),y=np.array(train_label).reshape(-1,1))
    pred = model.predict(test_feature)
    permut_importance = permutation_importance(model,test_feature,np.array(test_label),n_repeats=10)
    permute_importance_all_w_smiles_test.append(permut_importance.importances_mean)

    importance_all_dots_nn.append(permut_importance.importances)

    train_score = model.score(np.array(train_feature), np.array(train_label).reshape(-1, 1))
    val_score = model.score(np.array(val_feature), np.array(val_label).reshape(-1, 1))
    test_score = model.score(np.array(test_feature), np.array(test_label).reshape(-1, 1))
    test_score_all.append(test_score)


    print("The train score for this split is",train_score,"The val score for this split is",val_score,"The test score for this split is",test_score)
    prediction_mlp.append(pred)
    prediction_true_mlp.append(test_label)

mean_feature_importance_all_permute_smiles_test=np.mean(permute_importance_all_w_smiles_test,0)
sorted_feature_imporatnce_idx_permute_smiles_test=np.argsort(mean_feature_importance_all_permute_smiles_test[:-2])[::-1]
top_10_permute=sorted_feature_imporatnce_idx_permute_smiles_test[:10]
print('The top 10 most important substructure id from permutation importance is',top_10_permute)
last_10_permute = sorted_feature_imporatnce_idx_permute_smiles_test[-10:]
print('The last 10 most important substructure id from permutation importance is',last_10_permute)
print("mean_feature_importance_all_permute_smiles_test is",mean_feature_importance_all_permute_smiles_test)
'''
#数据集包含成熟作物的MAE
pred_data = model.predict(riped_data)
for i in range(3):
    error_=error(pred_data[i],riped_TF[i])
    print("Use model to predict Riped crop's TF error is", error_)
print(pred_data,riped_TF)
#数据集外成熟作物的MAE
pred_data = model.predict(extra_riped_data)
sum=0
for i in range(3):
    error_=error(pred_data[i],extra_TF_data[i])
    print("Use model to predict extra Riped crop's TF error is", error_)
print(pred_data,extra_TF_data)
#数据集中未成熟的作物
#数据集中未成熟作物
pred_data = model.predict(immature_data)
for i in range(3):
    error_ = error(pred_data[i], immature_TF[i])
    print("Use model to predict extra immature crop's TF error is", error_)
print(pred_data,immature_TF)
'''


ccmap = np.load('../data/chemical_group.npy')
ccmap_shuffle_property = []
for i in total_id:
    ccmap_shuffle_property.append(ccmap[i])

prediction_mlp_all = []
for l in prediction_mlp:
    for v in l:
        prediction_mlp_all.append(v)

prediction_true_mlp_all = []
for l in prediction_true_mlp:
    for v in l:
        prediction_true_mlp_all.append(v)

# 设置全局字体属性
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size('25')
font.set_weight('bold')
# 设置全局字体样式
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
plt.rcParams['font.weight'] = 'bold'   # 设置字体加粗
plt.rcParams['font.serif'] = ['Times New Roman'] # 设置全局字体样式
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)
sns.scatterplot(x=np.array(prediction_true_mlp).reshape(-1).tolist(),y=prediction_mlp_all,linewidth=0.5,hue=ccmap_shuffle_property,palette={1:'#6D042C',0:"#EECF63",2:"#3E7E6C"},s=35)
sns.lineplot(x=np.arange(-3.,3.0),y=np.arange(-3,3.0),color='r')
MAE=mean_absolute_error(prediction_true_mlp_all,prediction_mlp_all)
plt.text(-3,2,f"R-squared={round(np.mean(test_score_all),2)} MAE={round(MAE,2)}", ha='left', va='top',fontsize=25,fontproperties=font)
plt.xlabel('measured logTF',fontsize=30,fontproperties=font)
plt.ylabel('predicted logTF',fontsize=30,fontproperties=font)
# 加粗x轴
ax.spines['bottom'].set_linewidth(3)  # 底部x轴
ax.spines['top'].set_linewidth(3)  # 顶部x轴（如果需要）

# 加粗y轴
ax.spines['left'].set_linewidth(3)  # 左侧y轴
ax.spines['right'].set_linewidth(3)  # 右侧y轴（如果需要）
ax.tick_params(axis='both', labelsize=35,width=3)

handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles, ['A', 'B', 'C'],fontsize=30,loc="lower right")
frame = legend.get_frame()
frame.set_edgecolor('black')  # 设置边框颜色为黑色
frame.set_linewidth(2)  # 设置边框宽度为2.0
# if round(np.mean(test_score_all),2)>0.68:
#     break
# if time.time() - start_time > timeout:
#     print("\n时间到，程序退出")
#     break
print("r2={}".format(round(np.mean(test_score_all),2)))
plt.show()
