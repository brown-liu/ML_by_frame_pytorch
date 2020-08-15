import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)  # 1* 28 * 28=>1 * 24 * 24
                                                                        # max pooling => 12 * 12
        self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)  #    8 * 8
                                                                            # 4 * 4
        # self.conv3=nn.Conv2d(in_channels=12,out_channels=24,kernel_size=2) #3*3

        self.fc1=nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=60)
        self.out=nn.Linear(in_features=60,out_features=10)

        # input.shape = (n*n)
        # filter.shape=(f*f)
        # padding=p
        # stride = s
        # output_size = o
        # o = (n-f+2p)/s  + 1

    def forward(self, t):
        t=self.conv1(t)
        t=F.relu_(t)
        t=F.max_pool2d(t,kernel_size=2,stride=2)

        t=self.conv2(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size=2,stride=2)

        # t=self.conv3(t)
        # t=F.relu(t)
        # t=F.max_pool2d(t,kernel_size=2,stride=1)

        t=t.reshape(-1,12*4*4)
        t=self.fc1(t)
        t=F.relu(t)

        t=self.fc2(t)
        t=F.relu(t)

        t=self.out(t)
        t=F.softmax(t,dim=1)

        return t



train_set=torchvision.datasets.FashionMNIST(root='/Users/liubo/dataset/FashionMNIST',
                                            train=True,
                                            download=False,
                                            transform=transforms.Compose([transforms.ToTensor()]))
def get_accuracy(pred,labels):
    return pred.argmax(dim=1).eq(labels).sum().item()

cnnV1=Network()
train_loader=torch.utils.data.DataLoader(train_set,batch_size=100)
optimzer=torch.optim.Adam(cnnV1.parameters(),lr=0.01)

EPOCH=3



for epoch in tqdm(range(EPOCH)):
    total_run = 0
    total_correct = 0
    for batch in tqdm(train_loader):
        images,labels=batch
        preds=cnnV1(images)
        loss=F.cross_entropy(preds,labels)

        optimzer.zero_grad()

        loss.backward()
        optimzer.step()
        total_run+=100
        total_correct+=get_accuracy(preds, labels)


    print(f'total data unit = {total_run} \n'
          f'Accuracy = {round(total_correct / total_run, 3)}%\n'
          f'Total corect = {total_correct}')





torch.save(cnnV1.state_dict(), f'cnnV1params{str(time.time())}.pkl')
# model_object.load_state_dict(torch.load('params.pkl'))