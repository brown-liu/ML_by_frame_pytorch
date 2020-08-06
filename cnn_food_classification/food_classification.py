import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

from cnn_food_classification.CnnCore import cnnModel


def readimage(path,hasLabel):
    image_dir=sorted(os.listdir(path))
    x=np.zeros((len(image_dir),128,128,3),dtype=np.uint8)
    y=np.zeros(len(image_dir),dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img=cv2.imread(os.path.join(path,file))
        x[i,:,:]=cv2.resize(img,(128,128))

        if hasLabel:
            y[i]=int(file.split('_')[0])
        if hasLabel:
            return x,y
        else:
            return y

data_path=r'.\food-11'
print("Reading Dataset")
train_x,train_y=readimage(os.path.join(data_path,'training'),True)
print(f"Entries of training data = {len(train_x)}")
val_x,val_y=readimage(os.path.join(data_path,"validation"),True)
print(f"Entries of Validation data = {len(val_x)}")
test_x=readimage(os.path.join(data_path,"testing"),False)

train_transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

test_transform=transforms.Compose([

    transforms.ToPILImage(),
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self,x,y=None,transform=None):
        self.x=x
        self.y=y
        if y is not None:
            self.y=torch.LongTensor(y)
        self.transofm=transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X=self.x[index]
        if self.transofm is not None:
            X=self.transofm(X)
        if self.y is not None:
            Y=self.y[index]
            return X,Y
        else:return X

batch_size=128
train_set=ImgDataset(train_x,train_y,train_transform)
val_set=ImgDataset(val_x,val_y,test_transform)
train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
val_loader=DataLoader(val_set,batch_size=batch_size,shuffle=False)

model=cnnModel.classifier().cuda()
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
num_epoch=30

for epoch in range(num_epoch):
    epoch_start_time=time.time()
    train_acc=0.0
    train_loss=0.0
    val_acc=0.0
    val_loss=0.0

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred=model(data[0].cuda())
        batch_loss=loss(train_pred,data[1].cuda())
        batch_loss.backward()
        optimizer.step()
        train_acc+=np.sum(np.argmax(train_pred.cpu().data.numpy(),axis=1)==data[1].numpy())

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred=model(data[0].cuda())
            batch_loss=loss(val_pred,data[1].cuda())
            val_acc+=np.sum(np.argmax(val_pred.cpu().data.numpy(),axis=1)==data[1].numpy())
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

# increase amount of data for training
train_val_x=np.concatenate((train_x,val_x),axis=0)
train_val_y=np.concatenate((train_y,val_y),axis=0)
train_val_set=ImgDataset(train_val_x,train_val_y,train_transform)
train_val_loader=DataLoader(train_val_set,batch_size=batch_size,shuffle=True)

model_best=cnnModel.classifier().cuda()
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001) # optimizer 使用 Adam
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        #將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))

test_set=ImgDataset(test_x,transform=test_transform)
test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False)

model_best.eval()
prediction=[]
with torch.no_grad():
    for i,data in enumerate(test_loader):
        test_pred=model_best(data.cuda())
        test_label=np.argmax(test_pred.cpu().data.numpy(),axis=1)
        for y in test_label:
            prediction.append(y)

with open("predict.csv",'w') as f:
    f.write('Id,Category\n')
    for i, pred in enumerate(prediction):
        f.write(f'{i},{pred}')