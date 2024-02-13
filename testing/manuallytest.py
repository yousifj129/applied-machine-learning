import torch
from sklearn import datasets
from torch import optim 
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X,Y = datasets.make_regression(100,1,noise=20)

class LogisticModel(nn.Module):
    def __init__(self, inputsfeatures):
        super(LogisticModel,self).__init__()
        self.linear = nn.Linear(inputsfeatures,1)
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=533)
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0],1)
Y_test = Y_test.view(Y_test.shape[0],1)
model = LogisticModel(1)

crit = nn.BCELoss()

optimizer = optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    ypred = model(X_train)

    loss = crit(ypred,Y_train)

    loss.backward()

    optimizer.step()

    model.zero_grad()

    if (epoch+1) % 100 == 0:
        print(epoch, loss.item())

plt.plot(X_train,Y_train,'b.')
plt.plot(X_train,model(X_train).detach(),'r-')
plt.show()