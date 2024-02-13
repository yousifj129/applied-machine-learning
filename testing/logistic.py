import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
X,Y = bc.data, bc.target

n_samples,n_features = X.shape

print(n_samples,n_features)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=533)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0],1)
Y_test = Y_test.view(Y_test.shape[0],1)

class LogisticModel(nn.Module):
    def __init__(self, inputsfeatures):
        super(LogisticModel,self).__init__()
        self.linear = nn.Linear(inputsfeatures,1)
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    

model = LogisticModel(n_features)

criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    ypred = model(X_train)

    loss = criterion(ypred, Y_train)

    loss.backward()

    optimizer.step()

    model.zero_grad()
    if (epoch+1) % 10 == 0:
        print(epoch, loss.item())

with torch.no_grad():
    y_pred = model(X_test)
    ypredcls = y_pred.round()
    
    acc = ypredcls.eq(Y_test.view_as(ypredcls)).sum().item() / float(Y_test.shape[0])
    print(acc)

predicted = model(X).detach().numpy()

plt.plot(X_test, Y_test, 'ro')
plt.plot(X_test, predicted, 'b')

plt.show()