import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

training_data=np.load("training_data.npy",allow_pickle=True)
print(f'数据总长度{len(training_data)}条')

# plt.imshow(training_data[2333][0],cmap='gray')
# plt.show()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64,128, 5)

        x=torch.randn(50,50).view(-1,1,50,50)
        self._to_linear=None
        self.convs(x)
        self.fc1=nn.Linear(self._to_linear,512)
        self.fc2=nn.Linear(512,2)

    def convs(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        if self._to_linear is None:
            self._to_linear=x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x


    def forward(self,x):
        x=self.convs(x)
        x=x.view(-1,self._to_linear)
        x=F.relu(self.fc1(x))
        x= self.fc2(x)
        return F.softmax(x,dim=1)


device=torch.device("cuda:0")


cnn=CNN().to(device)


import torch.optim as optim
optimzer=optim.Adam(cnn.parameters(),  lr=0.00125)
loss_function=nn.MSELoss()

X=torch.tensor([i[0] for i in training_data]).view(-1,50,50)
X=X/255.0
y=torch.Tensor([i[1] for i in training_data])

VAL_PCT=0.1

val_size=int(len(X)*VAL_PCT)
print(f"取总训练数量的百分之{VAL_PCT*100} 为 {val_size}条")

X_train=X[:-val_size]
y_train=y[:-val_size]

X_test=X[-val_size:]
y_test=y[-val_size:]
print(f"训练数据总数 {len(X_train)} 条\n测试数据总数 {len(X_test)} 条")


def Accuracy(X_test, y_test):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(X_test))):
            real_class = torch.argmax(y_test[i]).to(device)
            cnn_out = cnn(X_test[i].view(-1, 1, 50, 50).to(device))[0]

            predicted_class = torch.argmax(cnn_out)

            if real_class == predicted_class:
                correct += 1
            total += 1
        result=round((correct * 100 / total),3)
        return result


BATCH_SIZE=100
EPOCHS=20
count=0
li_result=[]
li_count=[]
for epoch in range(EPOCHS):
    optimzer = optim.Adam(cnn.parameters(), lr=0.00125)
    loss_function = nn.MSELoss()
    for i in tqdm(range(0,len(X_train),BATCH_SIZE)):
        X_batch=X_train[i:i+BATCH_SIZE].view(-1,1,50,50).to(device)
        y_batch=y_train[i:i+BATCH_SIZE].to(device)

        cnn.zero_grad()

        predict=cnn(X_batch)
        loss=loss_function(predict,y_batch)
        loss.backward()
        optimzer.step()


    result=Accuracy(X_test,y_test)
    count+=1
    li_count.append(count)
    li_result.append(result)
    print(result)
    print(loss)

plt.plot(li_count,li_result)
plt.show()


