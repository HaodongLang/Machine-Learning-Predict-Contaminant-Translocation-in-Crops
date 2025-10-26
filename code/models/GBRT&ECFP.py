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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split,LeavePOut
from sklearn.model_selection import KFold #k折交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
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
from utility import Kfold,error
import scipy
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.early_stop import no_progress_loss
import shap
import matplotlib.font_manager as fm
import warnings
import creat_chemgrp
warnings.filterwarnings("ignore")
creat_chemgrp.creat_chemgrp()

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
inputDataX_scaler = np.concatenate((log_RCF,flipid),1)
inputDataX_for_importance=np.concatenate((inputDataX_scaler,FP),1)
# inputDataX_scaler=scipy.stats.mstats.zscore(inputDataX_for_importance)
inputDataX_scaler = inputDataX_for_importance
inputDataX_scaler[np.isnan(inputDataX_scaler)]=0
outputDatay=np.concatenate((pesticide_data.loc[:,"log TF"],PPCPs_data.loc[:,"log TF"],other_data.loc[:,"log TF"]),0).reshape(-1,1)
immature_data = inputDataX_scaler[:3]
immature_TF = outputDatay[:3]
immature_TF = immature_TF.ravel()
#数据集中成熟作物数据读取
condition = other_data["Citation"].str.contains('''Hexabromocyclododecanes in surface soil-maize system around
Baiyangdian Lake in North China: Distribution, enantiomer-specific''')
riped_data = other_data[condition]
riped_TF=riped_data.loc[:,"log TF"]
riped_TF=np.array(riped_TF)
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
total_id=np.arange(len(inputDataX_for_importance))
np.random.shuffle(total_id)
feature_importance_all_w_smiles=[]
permute_importance_all_w_smiles_test=[]

prediction_w_smiles = []
prediction_true_w_smiles = []
test_score_all_w_smiles = []
test_score_all = []

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
                  ,"max_features": hp.quniform("max_features",1,1026,1)
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

    test_score_all.append(test_score)
    permut_importance_test=permutation_importance(model,np.array(test_feature),np.array(test_label).reshape(-1,1),n_repeats=10)
    feature_importance_all_w_smiles.append(model.feature_importances_)      #all five splits feature importances
    permute_importance_all_w_smiles_test.append(permut_importance_test.importances_mean)
    prediction_gbrt.append(pred)
    prediction_true_gbrt.append(test_label)

    print("The train score for this split is",train_score,"The val score for this split is",val_score,"The test score for this split is",test_score)
print(model.predict(ECFP_α_HBCD))
print(model.predict(ECFP_b_HBCD))

mean_feature_importance_all_impurity_smiles=np.mean(feature_importance_all_w_smiles,0)
sorted_feature_imporatnce_idx_impurity_smiles=np.argsort(mean_feature_importance_all_impurity_smiles[:-2])[::-1]
top_10_impurity=sorted_feature_imporatnce_idx_impurity_smiles[:10]
print('The top 10 most important substructure id from impurity importance is',top_10_impurity)

mean_feature_importance_all_permute_smiles_test=np.mean(permute_importance_all_w_smiles_test,0)
sorted_feature_imporatnce_idx_permute_smiles_test=np.argsort(mean_feature_importance_all_permute_smiles_test[:-2])[::-1]
top_10_permute=sorted_feature_imporatnce_idx_permute_smiles_test[:10]
print('The top 10 most important substructure id from permutation importance is',top_10_permute)
last_10_permute = sorted_feature_imporatnce_idx_permute_smiles_test[-10:]
print('The last 10 most important substructure id from permutation importance is',last_10_permute)
print("mean_feature_importance_all_permute_smiles_test is",mean_feature_importance_all_permute_smiles_test)

'''
#数据集包含成熟作物
# riped_data = scipy.stats.mstats.zscore(riped_data)
# riped_data[np.isnan(riped_data)]=0
print(riped_data)
pred_data = model.predict(riped_data)
for i in range(3):
    error_=error(pred_data[i],riped_TF[i])
    print("Use model to predict Riped crop's TF error is",error_)
print(pred_data,riped_TF)

#数据集外成熟作物
pred_data = model.predict(extra_riped_data)
for i in range(3):
    error_=error(pred_data[i],extra_TF_data[i])
    print("Use model to predict extra Riped crop's TF error is", error_)
print(pred_data,extra_TF_data)

#数据集中未成熟作物
pred_data = model.predict(immature_data)
for i in range(3):
    error_ = error(pred_data[i], immature_TF[i])
    print("Use model to predict extra immature crop's TF error is", error_)
print(pred_data,immature_TF)
'''
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
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
plt.rcParams['font.weight'] = 'bold'   # 设置字体加粗
plt.rcParams['font.serif'] = ['Times New Roman'] # 设置全局字体样式
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)
# 加粗x轴
ax.spines['bottom'].set_linewidth(3)  # 底部x轴
ax.spines['top'].set_linewidth(3)  # 顶部x轴（如果需要）

# 加粗y轴
ax.spines['left'].set_linewidth(3)  # 左侧y轴
ax.spines['right'].set_linewidth(3)  # 右侧y轴（如果需要）
ax.tick_params(axis='both', labelsize=35,width=3)
# 设置全局字体属性
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size('25')

sns.scatterplot(x=np.array(prediction_true_gbrt_all).reshape(-1).tolist(),y=prediction_gbrt_all,linewidth=0.5,hue=ccmap_shuffle_property,palette={1:'#6D042C',0:"#EECF63",2:"#3E7E6C"},s=35)
sns.lineplot(x=np.arange(-3.,3.0),y=np.arange(-3,3.0),color='r')
MAE=mean_absolute_error(prediction_true_gbrt_all,prediction_gbrt_all)
plt.text(-3,2,"R-squared = %0.2f MAE = %0.2f" % (np.mean(test_score_all),MAE),ha='left',va='top',fontsize=25,fontproperties=font)
plt.xlabel('measured logTF',fontsize=30,fontproperties=font)
plt.ylabel('predicted logTF',fontsize=30,fontproperties=font)

handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles, ['A', 'B', 'C'],fontsize=30,loc="lower right")
frame = legend.get_frame()
frame.set_edgecolor('black')  # 设置边框颜色为黑色
frame.set_linewidth(2)  # 设置边框宽度为2.0
'''
modelPCA=PCA(n_components=2,copy=True)

modelECFP_GBRT=GradientBoostingRegressor(random_state=123,n_estimators=100,max_depth=2,loss='squared_error')
modelLM=LM.LinearRegression()
modelECFP_GBRT.fit(inputDataX_scaler,outputDatay)
modelLM.fit(outputDatay.values.reshape(341,1),modelECFP_GBRT.predict(inputDataX_scaler))
modelPCA.fit(bitData)

pcaData=modelPCA.fit_transform(bitData)[:,:]

#print(modelPCA.fit_transform(bitData)[:,:])
data=data.loc[:,'SMILES']
data=list(data)
fpList=[]
molList=[]
datalist=[]
for smile in data:
    molList.append(Chem.MolFromSmiles(smile))
for mol in molList:
    fpList.append(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024))
for fp in fpList:
    datalist.append(fp.ToBitString())
X=pd.DataFrame()
inputDataX_scaler=pd.DataFrame(inputDataX_scaler)
inputDataX_scaler.columns=['PCA1','PCA2','flipid','fom']

X=X.assign(**inputDataX_scaler)
X['log RCF-soil']=outputDatay

modelK.fit(X.values)test_score_all
label_pred = modelK.labels_
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
fig=plt.figure(figsize=(8,6))
ax0=fig.add_subplot(111)
ax0.scatter(x0.loc[:,'log RCF-soil'],modelECFP_GBRT.predict(x0.loc[:,['PCA1','PCA2','flipid','fom']]),c='r',label=1)
ax0.scatter(x1.loc[:,'log RCF-soil'],modelECFP_GBRT.predict(x1.loc[:,['PCA1','PCA2','flipid','fom']]),c='green',label=2)
ax0.scatter(x2.loc[:,'log RCF-soil'],modelECFP_GBRT.predict(x2.loc[:,['PCA1','PCA2','flipid','fom']]),c='y',label=3)
ax0.plot(outputDatay.values.reshape(341,1),modelLM.predict(outputDatay.values.reshape(341,1)),c='red',linestyle='-')
ax0.set_xlabel('measure logRCF')
ax0.set_ylabel('predicted logRCF')

ax0.legend(loc='lower right')
'''

plt.show()

