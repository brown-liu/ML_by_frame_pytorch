import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import time



start=time.time()
class RunBuilder():

    @staticmethod
    def get_runs(params):
        Run=namedtuple('Run',params.keys())
        runs=[]
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


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

IsWorkingOnMac=False

if IsWorkingOnMac:
    Localroot='/Users/liubo/dataset/FashionMNIST'
else:
    Localroot='D:\DataSets'

IsCudaAvailable=True
if IsCudaAvailable:
    device_select='cuda'
    print(f'Using Cuda By {torch.cuda.get_device_name()}')
else:
    device_select='cpu'
    print('Using CPU ONLY')
device=torch.device(device_select)


train_set=torchvision.datasets.FashionMNIST(
    root=Localroot,
    train=True,
    download=False,
    transform=transforms.Compose([transforms.ToTensor()]))

loader_temp=torch.utils.data.DataLoader(train_set)
data_temp=next(iter(loader_temp))
mean=data_temp[0].mean()
std=data_temp[0].std()

train_set_normal=torchvision.datasets.FashionMNIST(
         root=Localroot,
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
)

@torch.no_grad()
def get_accuracy(pred,labels):
    return pred.argmax(dim=1).eq(labels).sum().item()


params=OrderedDict(lr=[0.005,0.001],batch_size=[1000,5000,10000],shuffle=[True])
EPOCH=20


total_iternation=1
for keys in params:
    total_iternation*=len(params[keys])
total_iternation=total_iternation*EPOCH

runs=RunBuilder.get_runs(params)
cnnV1=Network().to(device)
count_iter=0
for lr,batch_size,shuffle in runs:

    comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle} EPOCH= {EPOCH}'
    tb = SummaryWriter(comment=comment)
    train_loader=torch.utils.data.DataLoader(train_set_normal,batch_size=batch_size,shuffle=shuffle)
    optimzer=torch.optim.Adam(cnnV1.parameters(),lr=lr)


    image001,label001=next(iter(train_loader))
    grid=torchvision.utils.make_grid(image001)

    tb.add_image('IMAGE',grid)
    tb.add_graph(cnnV1,image001.to(device))


    for epoch in range(EPOCH):
        total_run = 0
        total_correct = 0
        total_loss=0
        count_iter+=1
        print(f'Computing {count_iter}/{total_iternation}')
        for batch in tqdm(train_loader):
            images=batch[0].to(device)
            labels=batch[1].to(device)
            preds=cnnV1(images)
            loss=F.cross_entropy(preds,labels)

            optimzer.zero_grad()

            loss.backward()
            optimzer.step()
            total_run+=batch_size
            total_correct+=get_accuracy(preds, labels)
            total_loss+=loss

        tb.add_scalar('LOSS',total_loss,epoch)
        tb.add_scalar('Number Correct',total_correct,epoch)
        tb.add_scalar("Accuracy",(total_correct / total_run*100),epoch)

        tb.add_histogram("conv1.bias",cnnV1.conv1.bias,epoch)
        tb.add_histogram("conv1.weight",cnnV1.conv1.weight,epoch)
        tb.add_histogram('conv1.weight.grad',cnnV1.conv1.weight.grad,epoch)


        print(f' batch_size={batch_size} lr={lr} shuffle={shuffle} EPOCH= {EPOCH}')
        print(f'total data unit = {total_run} Accuracy = {round(total_correct / total_run*100, 3)}% Total corect = {total_correct}\n')

    tb.close()

end=time.time()
print(f'Total time used = {end-start}')


# torch.save(cnnV1.state_dict(), f'cnnV1params{str(time.time())}.pkl')
# model_object.load_state_dict(torch.load('params.pkl'))

# found max accuracy of 86.66% reach at EPOCH =10 lr= 0.001