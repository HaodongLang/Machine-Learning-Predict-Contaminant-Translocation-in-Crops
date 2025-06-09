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
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator

import creat_chemgrp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import train_test_split, LeavePOut, cross_validate
from sklearn.model_selection import KFold #k折交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
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
import scipy
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")
creat_chemgrp.creat_chemgrp()


pesticide_data = pd.read_excel("../data/data_final.xlsx",sheet_name="pesticide")
PPCPs_data = pd.read_excel("../data/data_final.xlsx",sheet_name="PPCPs")
other_data = pd.read_excel("../data/data_final.xlsx",sheet_name="other")

x_pesticide_data = pesticide_data.loc[:,["log RCF","flipid","MW（g/mol）","logKow"]].to_numpy().reshape(-1,4)
x_PPCPs_data=PPCPs_data.loc[:,["log RCF","flipid","MW（g/mol）","logKow"]].to_numpy().reshape(-1,4)
x_other_data=other_data.loc[:,["log RCF","flipid","MW（g/mol）","logKow"]].to_numpy().reshape(-1,4)


inputDataX_for_importance=np.concatenate((x_pesticide_data,x_PPCPs_data,x_other_data),0)
inputDataX_scaler=scipy.stats.mstats.zscore(inputDataX_for_importance)
inputDataX_scaler[np.isnan(inputDataX_scaler)]=0
outputDatay=np.concatenate((pesticide_data.loc[:,"log TF"],PPCPs_data.loc[:,"log TF"],other_data.loc[:,"log TF"]),0).reshape(-1,1)

train_split_index,test_split_index=Kfold(len(inputDataX_for_importance),5)
#print(np.array(train_split_index).shape,"\n",np.array(test_split_index).shape)
total_id=np.load("../data/sample_index.npy")
test_score_all = []
prediction_mlp = []
prediction_true_mlp = []
importance_all_dots_nn = []
feature_importance_all_w_smiles=[]
permute_importance_all_w_smiles_test=[]

fig = []
n_estimator = [(64,32),(128,64),(32,16),(128,64,32)]
batch_size = [64,128]

#贝叶斯优化
def opt_object(params):
    score = []
    model = MLPRegressor(hidden_layer_sizes = params["hidden_layer_sizes"]
              ,learning_rate_init = params["learning_rate_init"]
              ,batch_size = int(params["batch_size"])
              ,max_iter = int(params["max_iter"])
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




param_grid_simple = {'hidden_layer_sizes': hp.choice("hidden_layer_sizes",[(64,32),(128,64),(32,16),(128,64,32)])
                  ,"learning_rate_init": hp.quniform("learning_rate_init",0.0005,0.001,0.0001)
                  ,"batch_size": hp.choice("batch_size",[64,128])
                  ,"max_iter": hp.quniform("max_iter",1500,2000,100)
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

    # model = MLPRegressor(hidden_layer_sizes=(512,256), learning_rate_init=0.001, batch_size=64, max_iter=1000, random_state=1)

    model = MLPRegressor(hidden_layer_sizes=n_estimator[params_best["hidden_layer_sizes"]],
                         solver="adam", learning_rate_init=params_best["learning_rate_init"],
                         batch_size=batch_size[params_best["batch_size"]], max_iter=int(params_best["max_iter"]), random_state=1)
    model.fit(X=np.array(train_feature),y=np.array(train_label).reshape(-1,1))
    pred = model.predict(test_feature)
    permut_importance = permutation_importance(model,test_feature,np.array(test_label),n_repeats=10)
    importance_all_dots_nn.append(permut_importance.importances)

    train_score = model.score(np.array(train_feature), np.array(train_label).reshape(-1, 1))
    val_score = model.score(np.array(val_feature), np.array(val_label).reshape(-1, 1))
    test_score = model.score(np.array(test_feature), np.array(test_label).reshape(-1, 1))
    test_score_all.append(test_score)
    fig.append(plot_partial_dependence(model, np.array(train_feature),[0,1,2,3],feature_names=["log RCF","flipid","MW(g/mol)","logKow"],method = 'brute',kind = 'both',percentiles=(0,1)))


    print("The train score for this split is",train_score,"The val score for this split is",val_score,"The test score for this split is",test_score)
    prediction_mlp.append(pred)
    prediction_true_mlp.append(test_label)

mean_feature_importance_all_permute_smiles_test=np.mean(permute_importance_all_w_smiles_test,0)


plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
plt.rcParams['font.weight'] = 'bold'   # 设置字体加粗
plt.rcParams['font.serif'] = ['Times New Roman'] # 设置全局字体样式


figs,new_ax = plt.subplots(2,2,figsize=(18,18))


# 设置全局字体属性
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size('25')
font.set_weight('bold')
# 设置全局字体样式
plt.rc('font', family='Times New Roman')
plt.xticks(range(0, 101, 1))  # 设置x轴刻度，每隔1个单位一个刻度
plt.yticks(range(0, 101, 1))  # 设置y轴刻度，每隔1个单位一个刻度

for i in range(157):
    new_ax[0][0].plot(fig[2].pd_results[0]['values'][0],fig[2].pd_results[0]['individual'][0][i],color='salmon',alpha=0.05)
new_ax[0][0].xaxis.set_major_locator(MultipleLocator(1))
new_ax[0][0].yaxis.set_major_locator(MultipleLocator(1))
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
new_ax[0][1].xaxis.set_major_locator(MultipleLocator(1))
new_ax[0][1].yaxis.set_major_locator(MultipleLocator(1))
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
new_ax[1][0].xaxis.set_major_locator(MultipleLocator(1))
new_ax[1][0].yaxis.set_major_locator(MultipleLocator(1))
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
new_ax[1][1].xaxis.set_major_locator(MultipleLocator(1))
new_ax[1][1].yaxis.set_major_locator(MultipleLocator(1))
new_ax[1][1].plot(fig[2].pd_results[3]['values'][0],fig[2].pd_results[3]['average'][0],color='deepskyblue',label='average')
new_ax[1][1].tick_params(axis="both",labelsize=35,width=3)
new_ax[1][1].set_ylabel('Partial Dependence',fontsize=30,fontproperties=font)
new_ax[1][1].set_xlabel("logK$_{ow}$",fontsize=30,fontproperties=font)
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
for i in range(2):
    for tick in new_ax[0][i].yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
for i in range(2):
    for tick in new_ax[1][i].yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
for i in range(2):
    for tick in new_ax[0][i].xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
for i in range(2):
    for tick in new_ax[1][i].xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')

ccmap = np.load('../chemical_group.npy')

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


fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)
ax.tick_params(axis='both', labelsize=35,width=3)
sns.scatterplot(x=np.array(prediction_true_mlp).reshape(-1).tolist(),y=prediction_mlp_all,linewidth=0.5,hue=ccmap_shuffle_property,palette={1:'#6D042C',0:"#EECF63",2:"#3E7E6C"},s=35)
plt.xlabel('measured logTF',fontsize=30,fontproperties=font)
plt.ylabel('predicted logTF',fontsize=30,fontproperties=font)
plt.tick_params(axis='both', labelsize=35)
sns.lineplot(x=np.arange(-3.,3.0),y=np.arange(-3,3.0),color='r')
r_2=r2_score(prediction_true_mlp_all,prediction_mlp_all)
MAE=mean_absolute_error(prediction_true_mlp_all,prediction_mlp_all)
plt.text(-3.1,2,"R-squared = %0.2f MAE = %0.2f" % (np.mean(test_score_all),MAE),ha='left',va='top',fontsize=25,fontproperties=font)
font = FontProperties()
font.set_size('18')
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

plt.show()
