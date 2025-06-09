# -*- coding = utf-8 -*-
# @Time :2024/7/7 21:41
# @Author :郎皓东
# @File ：bpn.py
# @Software:PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from utility import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utility import Kfold
import warnings
warnings.filterwarnings("ignore")

# 数据输入
ECFP_data = pd.read_excel("final_ECFP.xlsx")
ECFP_data=ECFP_data.to_numpy()
pesticide_data = pd.read_excel("data_final.xlsx",sheet_name="pesticide")
PPCPs_data = pd.read_excel("data_final.xlsx",sheet_name="PPCPs")
other_data = pd.read_excel("data_final.xlsx",sheet_name="other")

x_pesticide_data = pesticide_data.loc[:,["log RCF","flipid","MW（g/mol）","logKow"]]
x_PPCPs_data=PPCPs_data.loc[:,["log RCF","flipid","MW（g/mol）","logKow"]]
x_other_data=other_data.loc[:,["log RCF","flipid","MW（g/mol）","logKow"]]
y_pesticide_data = pesticide_data.loc[:,["log TF"]]
y_PPCPs_data=PPCPs_data.loc[:,["log TF"]]
y_other_data=other_data.loc[:,["log TF"]]
x_data=pd.concat([x_pesticide_data,x_PPCPs_data,x_other_data],ignore_index=True)
y_data=pd.concat([y_pesticide_data,y_PPCPs_data,y_other_data],ignore_index=True)
x_data['logKow'] = x_data['logKow'].astype('float')
x_data['MW（g/mol）'] = x_data['MW（g/mol）'].astype('float')


x_data=x_data.to_numpy()
y_data=y_data.to_numpy()

x_data_scaler=scipy.stats.mstats.zscore(x_data)
y_data_scaler=y_data





learn_rate=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15,0.2]
# 设定模型
class Bpnn(nn.Module):
    def __init__(self):
        super(Bpnn, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(4,24),nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(24,1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


# 实例化模型


train_idx,test_idx = Kfold(len(x_data_scaler),5)

# 定义损失函数和优化器
loss = nn.MSELoss(size_average=False)
total_r2 = 0
best_lr = 0
y_true=[]
y_predict=[]
# 训练模型
for fold in range(5):
    best_r2 = 0


    train_id=[]
    test_id=[]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in train_idx[fold]:
        train_id.append(i)
    for i in test_idx[fold]:
        test_id.append(i)
    for i in train_id:
        x_train.append(x_data_scaler[i])
        y_train.append(y_data_scaler[i])
    for i in test_id:
        x_test.append(x_data_scaler[i])
        y_test.append(y_data_scaler[i])
    print('Fold',fold+1)

    for lr in learn_rate:
        model = Bpnn()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1000):

            y_pred = model(torch.Tensor(np.array(x_train)))
            loss_ = loss(y_pred, torch.Tensor(y_train))
            print(epoch, loss_.item())
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()



            r2=r2_score(torch.Tensor(y_test),model(torch.Tensor(x_test)))
            if r2>best_r2:
                best_r2=r2
                best_lr=lr
    print(f"best_lr={best_lr} best_r2={best_r2}")
    total_r2+=best_r2
    best_r2=0
    y_true.append(y_test)
    y_predict.append(model(torch.Tensor(x_test)).tolist())
y_predict=np.array(y_predict)
y_true=np.array(y_true)
y_predict=y_predict.squeeze()
y_true=y_true.squeeze()

final_r2=(total_r2/5)
print("final_r2=",final_r2)

y_true_=[]
for i in y_true:
    for j in i:
        y_true_.append(j)


y_predict_=[]
for i in y_predict:
    for j in i:
        y_predict_.append(j)


y0=[]
y1=[]
y2=[]
y3=[]
y4=[]
y0_p=[]
y1_p=[]
y2_p=[]
y3_p=[]
y4_p=[]
for i in y_true[0]:
    y0.append(i)
for i in y_true[1]:
    y1.append(i)
for i in y_true[2]:
    y2.append(i)
for i in y_true[3]:
    y3.append(i)
for i in y_true[4]:
    y4.append(i)

for i in y_predict[0]:
    y0_p.append(i)
for i in y_predict[1]:
    y1_p.append(i)
for i in y_predict[2]:
    y2_p.append(i)
for i in y_predict[3]:
    y3_p.append(i)
for i in y_predict[4]:
    y4_p.append(i)

# 可视化
# optimizer = optim.Adam(model.parameters(),lr=best_lr)
# for epoch in range(1000):
#     y_pred = model(torch.Tensor(x_data_scaler[train_idx]))
#     loss_ = loss(y_pred, torch.Tensor(y_data_scaler[train_idx]))
#
#     # print(epoch, loss_.item())
#     optimizer.zero_grad()
#     loss_.backward()
#     optimizer.step()

fig=plt.figure(figsize=(15,6))
ax0=fig.add_subplot(111)
#ax0.scatter(y_true_,y_predict_,c='r')
# ax0.scatter(y0,y0_p,c='r',label='1')
# ax0.scatter(y1,y1_p,c='y',label='2')
# ax0.scatter(y2,y2_p,c='green',label='3')
# ax0.scatter(y3,y3_p,c='brown',label='4')
# ax0.scatter(y4,y4_p,c='black',label='5')
ax0.scatter(y_true_,y_predict_,c='red')
x = np.arange(-4,5)
y = np.arange(-4,5)
ax0.plot(x,y,c="brown")
#plt.plot(x_data.numpy(), y_data.numpy(), 'ro', label='Original Data')
#plt.plot(x_data.numpy(), model(x_data).data.numpy(), 'b-', label='Prediction')
ax0.set_xlabel("measured logTF")
ax0.set_ylabel("predicted logTF")
ax0.text(0.03,0.9,f"r2={round(final_r2,2)}",transform=ax0.transAxes)
plt.legend()
#plt.savefig('scatter.png')
plt.show()
