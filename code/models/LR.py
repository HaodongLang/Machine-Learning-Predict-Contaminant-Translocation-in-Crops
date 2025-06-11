# -*- coding = utf-8 -*-
# @Time :2024/7/5 8:53
# @Author :郎皓东
# @File ：LR.py
# @Software:PyCharm
import scipy
import sklearn.linear_model as LM
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.font_manager as fm
warnings.filterwarnings("ignore",category=DeprecationWarning)
np.set_printoptions(threshold=np.inf)


pesticide_data=pd.read_excel("../data/data_final.xlsx",sheet_name="pesticide")
pesticide_compound=pesticide_data.loc[:,"Compound"].unique().tolist()
PPCPs_data=pd.read_excel("../data/data_final.xlsx",sheet_name="PPCPs")
PPCPs_compound=PPCPs_data.loc[:,"Compound"].unique().tolist()
other_data=pd.read_excel("../data/data_final.xlsx",sheet_name="other")
other_compound=other_data.loc[:,"Compound"].unique().tolist()

inputX_pesticide=pesticide_data.loc[:,["log RCF","MW（g/mol）","logKow","flipid"]]
inputX_PPCPs=PPCPs_data.loc[:,["log RCF","MW（g/mol）","logKow","flipid"]]
inputX_other=other_data.loc[:,["log RCF","MW（g/mol）","logKow","flipid"]]
inputX=pd.concat([inputX_pesticide,inputX_PPCPs,inputX_other],ignore_index=True)
inputX=scipy.stats.mstats.zscore(inputX)
outputy_pesticide=pesticide_data.loc[:,["log TF"]]
outputy_PPCPs=PPCPs_data.loc[:,["log TF"]]
outputy_other=other_data.loc[:,["log TF"]]
outputy=pd.concat([outputy_pesticide,outputy_PPCPs,outputy_other],ignore_index=True)
X=pd.DataFrame()
X=X.assign(**inputX)
X['logTF']=outputy
modelK=KMeans(n_clusters=3)
modelLM=LM.LinearRegression()
modelLM.fit(inputX,outputy)
pred_y=modelLM.predict(inputX)
modelK.fit(X.values)
pred_label=modelK.labels_
x0=X[pred_label==0]
x1=X[pred_label==1]
x2=X[pred_label==2]


total_id = np.load("../data/sample_index.npy")

ccmap = np.load('../chemical_group.npy')
print(ccmap)
ccmap_shuffle_property = []
for i in total_id:
    ccmap_shuffle_property.append(ccmap[i])

plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
plt.rcParams['font.weight'] = 'bold'   # 设置字体加粗
plt.rcParams['font.serif'] = ['Times New Roman'] # 设置全局字体样式
font = FontProperties()
font.set_size('18')
font.set_family('serif')
font.set_name('Times New Roman')
font.set_weight('bold')
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111)
ax.set_xlabel('measured logTF',fontsize=30,fontproperties=font)
ax.set_ylabel('predicted logTF',fontsize=30,fontproperties=font)
sns.scatterplot(x=np.array(outputy.values.reshape(-1).tolist()).reshape(-1).tolist(),y=pred_y.reshape(-1).tolist(),linewidth=0.5,hue=ccmap_shuffle_property,palette={1:'#6D042C',0:"#EECF63",2:"#3E7E6C"},s=35)

sns.lineplot(x=np.arange(-3.,3.0),y=np.arange(-3,3.0),color='r')
r2=r2_score(outputy,pred_y)
MAE = np.abs(cross_val_score(modelLM,outputy,pred_y,cv=5,scoring='neg_mean_absolute_error')).mean()
plt.text(-3.1,2,"R-squared = %0.2f MAE = %0.2f" % (r2,MAE),ha='left',va='top',fontsize=25,fontproperties=font)


# 加粗x轴和y轴的刻度标签
plt.xticks(fontweight='bold')  # 加粗x轴的刻度标签
plt.yticks(fontweight='bold')  # 加粗y轴的刻度标签
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
# 获取画布的尺寸，单位为英寸
fig_size = fig.get_size_inches()
plt.savefig("LR.png")
plt.show()
