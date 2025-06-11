# -*- coding = utf-8 -*-
# @Time :2024/7/5 8:29
# @Author :郎皓东
# @File ：ECFP&PCA.py
# @Software:PyCharm
import numpy as np
from matplotlib import patches
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

pesticide_data = pd.read_excel("../data/data_final.xlsx",sheet_name="pesticide")
PPCPs_data = pd.read_excel("../data/data_final.xlsx",sheet_name="PPCPs")
other_data = pd.read_excel("../data/data_final.xlsx",sheet_name="other")

SMILES_pesticide=pesticide_data.loc[:,"SMILES"]
SMILES_PPCPs=PPCPs_data.loc[:,"SMILES"]
SMILES_other=other_data.loc[:,"SMILES"]
SMILES=np.concatenate((SMILES_pesticide,SMILES_PPCPs,SMILES_other),0)

Compound_pesticide=pesticide_data.loc[:,"Compound"]
Compound_PPCPs=PPCPs_data.loc[:,"Compound"]
Compound_other=other_data.loc[:,"Compound"]
Compound = np.concatenate((Compound_pesticide,Compound_PPCPs,Compound_other),0)

FP=[]
for SMILE in SMILES:
    info = {}
    mol=Chem.MolFromSmiles(SMILE)
    fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,bitInfo=info)
    FP.append(fp)
FP=np.array(FP)
FP=pd.DataFrame(FP)
#FP.to_excel("bit_data.xlsx")
data=pd.read_excel("../data/bit_data.xlsx")
modelPCA=PCA(n_components=2,copy=True,random_state=111)
modelK=KMeans(n_clusters=3,random_state=111)
FP=modelPCA.fit_transform(FP)
modelK.fit(FP)
label_pred=modelK.labels_
centers = modelK.cluster_centers_
seq_row = np.arange(225).reshape(-1,1)
FP = np.c_[FP,seq_row]
x0=FP[label_pred==0]
x0 = np.unique(x0[:,:2],axis=0)
data = pd.DataFrame(x0)
data.to_excel('output.xlsx', index=False)
x1=FP[label_pred==1]
x2 = np.unique(x1[:,:2],axis=0)

x2=FP[label_pred==2]
x2 = np.unique(x2[:,:2],axis=0)

plt.rcParams['font.family'] = 'serif'  # 使用衬线字体
plt.rcParams['font.weight'] = 'bold'   # 设置字体加粗
# print(set(list(Compound[list(map(int,list(x0[:,2])))])))
# print(set(list(Compound[list(map(int,list(x1[:,2])))])))
# print(set(list(Compound[list(map(int,list(x2[:,2])))])))

font = FontProperties()
font.set_family('serif')
font.set_weight('bold')
font.set_name('Times New Roman')
font.set_size('30')
# 设置全局字体样式
plt.rc('font', family='Times New Roman')

fig=plt.figure(figsize=(15,10))
ax0=fig.add_subplot(111)
ax0.xaxis.set_major_locator(MultipleLocator(1))
ax0.scatter(x0[:,0],x0[:,1],c='r',label="A")
ax0.scatter(x1[:,0],x1[:,1],c='g',label="B")
ax0.scatter(x2[:,0],x2[:,1],c='y',label="C")
ax0.tick_params(axis='both', labelsize=40,width=3)
# 加粗x轴
ax0.spines['bottom'].set_linewidth(3)  # 底部x轴
ax0.spines['top'].set_linewidth(3)  # 顶部x轴（如果需要）

# 加粗y轴
ax0.spines['left'].set_linewidth(3)  # 左侧y轴
ax0.spines['right'].set_linewidth(3)  # 右侧y轴（如果需要）
plt.xlabel("PCA1",fontsize=35,fontproperties=font)
plt.ylabel("PCA2",fontsize=35,fontproperties=font)
font = FontProperties()
font.set_size('22')
legend = ax0.legend(loc="lower left",fontsize=30)
frame = legend.get_frame()
frame.set_edgecolor('black')  # 设置边框颜色为黑色
frame.set_linewidth(2)  # 设置边框宽度为2.0


plt.savefig("PCA.png")
plt.show()


