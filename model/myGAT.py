import torch
import math
import pandas as pd
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
import ssl
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

data1=pd.read_csv("r1_onehot.csv",index_col=0)
data2=pd.read_csv("d1_onehot.csv",index_col=0)
data3=pd.read_csv("l1_onehot.csv",index_col=0)
data1=data1.iloc[:,1:]
data2=data2.iloc[:,1:]
data3=data3.iloc[:,1:]
my_array1 = np.array(data1)
my_array1=scale(X=my_array1,axis=0,with_mean=True,with_std=True,copy=True)
x1 = torch.tensor(my_array1, dtype=torch.float)
my_array2 = np.array(data2)
my_array2=scale(X=my_array2,axis=0,with_mean=True,with_std=True,copy=True)
x2 = torch.tensor(my_array2, dtype=torch.float)
my_array3 = np.array(data3)
my_array3=scale(X=my_array3,axis=0,with_mean=True,with_std=True,copy=True)
x3 = torch.tensor(my_array3, dtype=torch.float)
datar1=pd.read_csv("ahmadr1.csv")
datad1=pd.read_csv("d1_onehot.csv",index_col=0)
datal1=pd.read_csv("ahmadl1.csv")
datad1=pd.DataFrame(cosine_similarity(datad1.iloc[:,1:]))

datar1[datar1<15000000000]=1
datad1[datad1>0.996974]=1
datal1[datal1<2188000000]=1
datar1[datar1!=1]=0
datad1[datad1!=1]=0
datal1[datal1!=1]=0
dr1=datar1
dr2=datad1
dr3=datal1

dr1=np.array(datar1)
edge_index1=np.array(np.where(dr1 == 1))
dr2=np.array(datad1)
edge_index2=np.array(np.where(dr2 == 1))
dr3=np.array(datal1)
edge_index3=np.array(np.where(dr3 == 1))
edge_index1=torch.tensor(edge_index1, dtype=torch.long)
edge_index2=torch.tensor(edge_index2, dtype=torch.long)
edge_index3=torch.tensor(edge_index3, dtype=torch.long)

m=pd.read_csv("binary.csv",index_col=0)
y=np.array(m['y_or_n'])
y = torch.tensor(y,dtype=torch.float)

#mask
train_mask = torch.zeros(y.size(0), dtype=torch.bool)
for i in range(int(y.max()) + 1):
    p=int(((y == i).nonzero(as_tuple=False).shape[0])*0.7)
    train_mask[(y == i).nonzero(as_tuple=False)[0:p]] = True
remaining=(~train_mask).nonzero(as_tuple=False).view(-1)
remaining=remaining[torch.randperm(remaining.size(0))]
test_mask = torch.zeros(y.size(0), dtype=torch.bool)
test_mask.fill_(False)
test_mask[remaining[:]]=True

dataset1 = Data(x=x1, edge_index=edge_index1, y=y, train_mask=train_mask, test_mask=test_mask)
dataset2 = Data(x=x2, edge_index=edge_index2, y=y, train_mask=train_mask, test_mask=test_mask)
dataset3 = Data(x=x3, edge_index=edge_index3, y=y, train_mask=train_mask, test_mask=test_mask)

data_all = {'data1': dataset1, 'data2':dataset2, 'data3':dataset3}
data_all={key:data_all[key].cuda() for key in data_all}

class Net_all(torch.nn.Module):
    def __init__(self):
        super(Net_all, self).__init__()
        self.gat1_1 = GATConv(dataset1.num_node_features, 8, 9, dropout=0.8)
        #self.gat1_2 = GATConv(64, 8, 8, dropout=0.8)
        
        self.gat2_1 = GATConv(dataset2.num_node_features, 8, 9, dropout=0.8)
        #self.gat2_2 = GATConv(64, 8, 8, dropout=0.8)

        self.gat3_1 = GATConv(dataset3.num_node_features, 8, 9, dropout=0.8)
        #self.gat3_2 = GATConv(64, 8, 8, dropout=0.8)
        self.re = nn.ReLU()
        self.fnn = nn.Linear(216, 2)

    def forward(self, data):
        data1 = data['data1']
        data2 = data['data2']
        data3 = data['data3']

        x1 = self.gat1_1(data1.x, data1.edge_index)
        #x1 = self.gat1_2(x1, data1.edge_index)
        x2 = self.gat2_1(data2.x, data2.edge_index)
        #x2 = self.gat2_2(x2, data2.edge_index)
        x3 = self.gat3_1(data3.x, data3.edge_index)
        #x3 = self.gat3_2(x3, data3.edge_index)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        x = torch.cat((x1,x2, x3), 1)
        # print(x.shape)
        x = self.re(x)
        x = self.fnn(x)
        # x = x.view(data1.x.shape[0])
        return F.softmax(x, dim=1)

# torch.cuda.empty_cache()
data = data_all
model = Net_all().to('cuda')
# data = dataset.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()
loss_fn = torch.nn.MSELoss()
train_iterations = []
train_loss = []
for epoch in range(2000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data['data1'].train_mask], data['data1'].y[data['data1'].train_mask].long())
    loss.backward()
    #print(loss)
    optimizer.step()
    train_loss.append(loss.item())
    train_iterations.append(epoch+1)
    print(f"epoch:{epoch+1}, loss:{loss.item()}")
model.eval()
#pred = model(data).max(dim=1).indices
#correct = int(pred[data['data1'].test_mask].eq(data['data1'].y[data['data1'].test_mask]).sum().item())
#acc = correct / int(data['data1'].test_mask.sum())
#print('Accuracy:{:.4f}'.format(acc))

pred = model(data).max(dim=1).indices
correct = int(pred[data['data1'].test_mask].eq(data['data1'].y[data['data1'].test_mask]).sum().item())
acc = correct / int(data['data1'].test_mask.sum())
print('Accuracy:{:.4f}'.format(acc))

from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
host = host_subplot(111)

plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window
par1 = host.twinx()
# set labels
host.set_xlabel("iterations")
host.set_ylabel("log loss")
p1, = host.plot(train_iterations, train_loss, label="training log loss")

y_t = data['data1'].y[data['data1'].train_mask].cpu().numpy()
predr = pred[data['data1'].train_mask].cpu().numpy()

print("accuracy_score:", accuracy_score(y_t, predr))
print("ROC AUC score:", roc_auc_score(y_t, predr))
p = precision_score(y_t, predr, average='binary')
r = recall_score(y_t, predr, average='binary')
f1score = f1_score(y_t, predr, average='binary')
print("precision_score:",p)
print("recall_score:",r)
print("f1_score:",f1score)
print("matrics£º",confusion_matrix(y_t, predr))

pred.cpu().numpy()
m['pred']=pred.cpu().numpy()
m.to_csv("pre0322.csv")